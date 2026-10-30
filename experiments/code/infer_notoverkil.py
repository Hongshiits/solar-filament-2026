"""Validate notoverkil's public fold0 models on OUR honest fold0 val split.

Two detectors:
  * p1-maskrcnn1536-long-f0 / last.pt  (torchvision maskrcnn_resnet50_fpn_v2, 1536)
  * p1-yolo11m1536-long-f0 / last.pt   (ultralytics yolo11m-seg, 1536)
Configs: work/scan_assets/notoverkil/<run>/config.yaml

Inference spec (from config):
  maskrcnn: resize 2048->1536 INTER_AREA, 3-channel repeated grayscale /255,
            score_thr 0.3, mask_thr 0.5, min_area 150, max_area 50000, max_det 100.
  yolo    : ultralytics predict imgsz=1536, conf 0.2, iou 0.7, min_area 150,
            max_area 50000, max_det 60.
Masks are mapped back to 2048 and greedily de-overlapped; official pooled PQ is
computed with src/pq_metric.py against every annotator record of fold0.

Usage:
    python src/infer_notoverkil.py --mode diagnose
    python src/infer_notoverkil.py --mode run --limit 5
    python src/infer_notoverkil.py --mode run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pq_metric as P  # noqa: E402

H = W = 2048
ASSET_DEFAULT = Path(r"C:\Users\ptcu_\filament_assets\downloads\notoverkil")
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_JSON = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
FOLDS = ROOT / "work" / "cache" / "folds.json"


def _set_data_paths(data_root: str | None, folds_json: str | None, train_json: str | None = None) -> None:
    """Allow running on Kaggle (data mounted under /kaggle/input) or locally."""
    global DATA, TRAIN_JSON, FOLDS
    if data_root:
        root = Path(data_root)
        if (root / "train").exists():
            DATA = root
        else:
            candidates = sorted(root.glob("**/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
            if not candidates:
                candidates = sorted(root.glob("**/train_images"))
            if candidates:
                DATA = candidates[0].parent.parent if candidates[0].name.endswith(".json") else candidates[0].parent
    if train_json:
        TRAIN_JSON = Path(train_json)
    else:
        TRAIN_JSON = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if folds_json:
        FOLDS = Path(folds_json)

MASK_RCNN_CFG = {
    "image_size": 1536,
    "score_threshold": 0.3,
    "mask_threshold": 0.5,
    "minimum_area": 150,
    "maximum_area": 50000,
    "maximum_detections": 100,
}
YOLO_CFG = {
    "image_size": 1536,
    "score_threshold": 0.2,
    "nms_iou": 0.7,
    "mask_threshold": 0.5,
    "minimum_area": 150,
    "maximum_area": 50000,
    "maximum_detections": 60,
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ----------------------------------------------------------------------------
# shared entry helpers (bbox-local masks for fast dedupe / painting)
# ----------------------------------------------------------------------------
def entry_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    return {"x0": x0, "y0": y0, "m": mask[y0 : int(ys.max()) + 1, x0 : int(xs.max()) + 1].astype(np.uint8)}


def entry_area(entry) -> int:
    return int(entry["m"].sum())


def entry_to_full(entry) -> np.ndarray:
    full = np.zeros((H, W), dtype=bool)
    full[entry["y0"] : entry["y0"] + entry["m"].shape[0], entry["x0"] : entry["x0"] + entry["m"].shape[1]] = entry["m"] > 0
    return full


def entry_iou(a, b) -> float:
    x0 = max(a["x0"], b["x0"])
    y0 = max(a["y0"], b["y0"])
    x1 = min(a["x0"] + a["m"].shape[1], b["x0"] + b["m"].shape[1])
    y1 = min(a["y0"] + a["m"].shape[0], b["y0"] + b["m"].shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    va = a["m"][y0 - a["y0"] : y1 - a["y0"], x0 - a["x0"] : x1 - a["x0"]]
    vb = b["m"][y0 - b["y0"] : y1 - b["y0"], x0 - b["x0"] : x1 - b["x0"]]
    inter = int(np.logical_and(va, vb).sum())
    union = entry_area(a) + entry_area(b) - inter
    return inter / union if union > 0 else 0.0


def subtract_entry(target, other):
    x0 = max(target["x0"], other["x0"])
    y0 = max(target["y0"], other["y0"])
    x1 = min(target["x0"] + target["m"].shape[1], other["x0"] + other["m"].shape[1])
    y1 = min(target["y0"] + target["m"].shape[0], other["y0"] + other["m"].shape[0])
    if x1 <= x0 or y1 <= y0:
        return target
    view = target["m"][y0 - target["y0"] : y1 - target["y0"], x0 - target["x0"] : x1 - target["x0"]]
    other_view = other["m"][y0 - other["y0"] : y1 - other["y0"], x0 - other["x0"] : x1 - other["x0"]]
    overlap = np.logical_and(view, other_view)
    if not np.any(overlap):
        return target
    updated = target["m"].copy()
    updated[y0 - target["y0"] : y1 - target["y0"], x0 - target["x0"] : x1 - target["x0"]][overlap] = 0
    ys, xs = np.where(updated)
    if len(xs) == 0:
        return None
    bx0, by0 = int(xs.min()), int(ys.min())
    bx1, by1 = int(xs.max()) + 1, int(ys.max()) + 1
    return {"x0": target["x0"] + bx0, "y0": target["y0"] + by0, "m": np.ascontiguousarray(updated[by0:by1, bx0:bx1], dtype=np.uint8)}


def resolve_instances(items: list[dict], min_area: int, max_area: int, max_det: int):
    """items: {'mask': 2048 bool, 'score': float, 'source': str} -> non-overlapping entries."""
    items = sorted(items, key=lambda item: -item["score"])
    entries = []
    for item in items:
        entry = entry_from_mask(item["mask"])
        if entry is None:
            continue
        area = entry_area(entry)
        if area < min_area or area > max_area:
            continue
        updated = entry
        for previous in entries:
            updated = subtract_entry(updated, previous)
            if updated is None:
                break
        if updated is None or entry_area(updated) < min_area:
            continue
        item["entry"] = updated
        entries.append(updated)
        if len(entries) >= max_det:
            break
    return entries, [item for item in items if "entry" in item]


# ----------------------------------------------------------------------------
# mask r-cnn
# ----------------------------------------------------------------------------
def load_maskrcnn_state(path: Path):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    meta = {}
    state = payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "ema", "module", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                meta = {k: v for k, v in payload.items() if k not in (key,)}
                break
    return state, meta


def build_maskrcnn(num_classes: int, in_channels: int, size: int):
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=size, max_size=size)
    if in_channels != 3:
        old = model.backbone.body.conv1
        new = torch.nn.Conv2d(
            in_channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=old.bias is not None
        )
        model.backbone.body.conv1 = new
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    in_feats_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feats_mask, 256, num_classes)
    return model


def diagnose_maskrcnn(path: Path):
    if not path.exists():
        log(f"missing maskrcnn weights: {path}")
        return None
    state, meta = load_maskrcnn_state(path)
    keys = list(state.keys())
    log(f"maskrcnn ckpt keys: {len(keys)}; meta keys: {list(meta.keys())[:6]}")
    for probe in (
        "backbone.body.conv1.weight",
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.bbox_pred.weight",
        "roi_heads.mask_predictor.conv5_mask.weight",
        "roi_heads.mask_predictor.mask_fcn_logits.weight",
        "rpn.head.conv.weight",
    ):
        if probe in state:
            log(f"  {probe}: {tuple(state[probe].shape)}")
    conv1 = state.get("backbone.body.conv1.weight")
    cls = state.get("roi_heads.box_predictor.cls_score.weight")
    num_classes = int(cls.shape[0]) if cls is not None else 2
    in_channels = int(conv1.shape[1]) if conv1 is not None else 3
    model = build_maskrcnn(num_classes, in_channels, MASK_RCNN_CFG["image_size"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"  load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        log(f"  missing sample: {missing[:5]}")
    if unexpected[:5]:
        log(f"  unexpected sample: {unexpected[:5]}")
    return {"num_classes": num_classes, "in_channels": in_channels, "missing": len(missing), "unexpected": len(unexpected)}


def infer_maskrcnn(model, device, image_paths: list[Path], limit: int = 0, size_override: int | None = None) -> dict[str, list[dict]]:
    cfg = dict(MASK_RCNN_CFG)
    if size_override:
        cfg["image_size"] = int(size_override)
    size = cfg["image_size"]
    output: dict[str, list[dict]] = {}
    if limit:
        image_paths = image_paths[:limit]
    model.eval()
    for number, path in enumerate(image_paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        if image.shape != (H, W):
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().unsqueeze(0).repeat(3, 1, 1) / 255.0
        with torch.no_grad():
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model([tensor.to(device)])[0]
        scores = out["scores"].detach().cpu().numpy().astype(float)
        masks = out["masks"].detach().cpu().numpy()[:, 0]  # (N, size, size) already thresholded at 0.5
        items = []
        for score, mask in zip(scores, masks):
            if score < cfg["score_threshold"]:
                continue
            mask_full = cv2.resize((mask > cfg["mask_threshold"]).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            area = int(mask_full.sum())
            if area < cfg["minimum_area"] or area > cfg["maximum_area"]:
                continue
            items.append({"mask": mask_full, "score": float(score), "source": "maskrcnn"})
        stage = time.time()
        entries, kept = resolve_instances(items, cfg["minimum_area"], cfg["maximum_area"], cfg["maximum_detections"])
        output[path.stem] = kept
        log(f"maskrcnn {number}/{len(image_paths)}: raw={len(items)} kept={len(kept)} forward+post={time.time()-stage:.1f}s")
    return output


# ----------------------------------------------------------------------------
# yolo
# ----------------------------------------------------------------------------
def load_yolo_checkpoint(path: Path, converted_dir: Path | None = None):
    """Load a notoverkil YOLO checkpoint; convert their custom format if needed.

    Their last.pt is {"ema": OrderedDict, "config": {...}, "architecture": {...}}
    which ultralytics rejects. We rebuild the SegmentationModel from
    ckpt["architecture"], load ckpt["ema"] (0 missing keys) and save a standard
    ultralytics checkpoint that YOLO() can consume.
    """
    from ultralytics import YOLO

    try:
        model = YOLO(str(path))
        log(f"yolo: loaded directly ({path.name})")
        return model
    except Exception as error:  # noqa: BLE001
        log(f"yolo: direct load failed ({str(error)[:160]}); rebuilding from architecture")

    from ultralytics.nn.tasks import SegmentationModel

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not (isinstance(payload, dict) and "architecture" in payload and "ema" in payload):
        raise RuntimeError("unsupported notoverkil yolo checkpoint format")
    architecture = dict(payload["architecture"])
    architecture["task"] = "segment"
    model = SegmentationModel(architecture, ch=int(architecture.get("ch", 3)), nc=int(architecture.get("nc", 1)), verbose=False)
    missing, unexpected = model.load_state_dict(payload["ema"], strict=False)
    log(f"yolo: rebuilt from architecture, missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        raise RuntimeError(f"yolo rebuild missing keys: {missing[:5]}")
    model.task = "segment"
    model.names = {0: "filament"}
    model.args = {"task": "segment"}
    model.eval()
    target_dir = converted_dir or path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    converted = target_dir / f"{path.stem}_ultralytics_seg.pt"
    torch.save(
        {
            "model": model,
            "epoch": 0,
            "version": "8.4.191",
            "date": time.strftime("%Y-%m-%d"),
            "task": "segment",
            "train_args": {"task": "segment", "imgsz": 1536},
        },
        converted,
    )
    log(f"yolo: converted checkpoint -> {converted}")
    return YOLO(str(converted))


def diagnose_yolo(path: Path):
    if not path.exists():
        log(f"missing yolo weights: {path}")
        return None
    try:
        model = load_yolo_checkpoint(path)
        info = {"task": getattr(model, "task", None), "names": getattr(model, "names", None)}
        log(f"yolo loaded: task={info['task']} names={info['names']}")
        return info
    except Exception as error:  # noqa: BLE001
        log(f"yolo load failed: {error}")
        return None


def infer_yolo(model, device, image_paths: list[Path], limit: int = 0) -> dict[str, list[dict]]:
    cfg = YOLO_CFG
    if limit:
        image_paths = image_paths[:limit]
    output: dict[str, list[dict]] = {}
    for number, path in enumerate(image_paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        if image.shape != (H, W):
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        result = model.predict(
            source=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
            imgsz=cfg["image_size"],
            conf=cfg["score_threshold"],
            iou=cfg["nms_iou"],
            agnostic_nms=True,
            retina_masks=True,
            max_det=cfg["maximum_detections"],
            verbose=False,
            device="cuda:0" if device.type == "cuda" else "cpu",
        )[0]
        items = []
        if result.masks is not None and result.boxes is not None and len(result.masks.data):
            masks = result.masks.data.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy().astype(float)
            for score, mask in zip(scores, masks):
                mask = mask.astype(np.uint8)
                if mask.shape != (H, W):
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask.astype(bool)
                area = int(mask_bool.sum())
                if area < cfg["minimum_area"] or area > cfg["maximum_area"]:
                    continue
                items.append({"mask": mask_bool, "score": float(score), "source": "yolo"})
        stage = time.time()
        entries, kept = resolve_instances(items, cfg["minimum_area"], cfg["maximum_area"], cfg["maximum_detections"])
        output[path.stem] = kept
        log(f"yolo {number}/{len(image_paths)}: raw={len(items)} kept={len(kept)} forward+post={time.time()-stage:.1f}s")
    return output


# ----------------------------------------------------------------------------
# evaluation / fusion
# ----------------------------------------------------------------------------
def to_pred_df(predictions: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for stem in sorted(predictions):
        items = sorted(predictions[stem], key=lambda item: -item["score"])
        for index, item in enumerate(items, 1):
            full = entry_to_full(item["entry"]) if "entry" in item else item["mask"]
            counts = mask_utils.encode(np.asfortranarray(full.astype(np.uint8)))["counts"]
            rows.append({"filament_id": f"{stem}_{index}", "segmentation_rle": counts.decode("ascii")})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def fuse(predictions_a: dict[str, list[dict]], predictions_b: dict[str, list[dict]], min_area: int, max_area: int, max_det: int):
    fused = {}
    for stem in sorted(set(predictions_a) | set(predictions_b)):
        items = list(predictions_a.get(stem, [])) + list(predictions_b.get(stem, []))
        entries, kept = resolve_instances(items, min_area, max_area, max_det)
        fused[stem] = kept
    return fused


def fold0_val_stems() -> list[str]:
    folds = json.load(open(FOLDS, encoding="utf-8"))
    return sorted(folds["folds"]["0"])


def resolve_image_paths(stems: list[str]) -> list[Path]:
    images = DATA / "train" / "train_images"
    return [images / f"{stem}.jpeg" for stem in stems]


def evaluate(predictions: dict[str, list[dict]], label: str, gt_df, stems: list[str]) -> dict:
    pred_df = to_pred_df({stem: predictions.get(stem, []) for stem in stems})
    started = time.time()
    result = P.evaluate(gt_df, pred_df)
    report = {key: value for key, value in result.items() if key not in ("per_image", "tp_ious")}
    rows = len(pred_df)
    log(f"EVAL {label}: PQ={report['pq']:.6f} SQ={report['sq']:.6f} RQ={report['rq']:.6f} TP={report['tp']} FP={report['fp']} FN={report['fn']} rows={rows} ({time.time()-started:.0f}s)")
    return {"label": label, "pq": report["pq"], "sq": report["sq"], "rq": report["rq"], "tp": report["tp"], "fp": report["fp"], "fn": report["fn"], "n_rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="run", choices=["diagnose", "run"])
    parser.add_argument("--asset-dir", default=str(ASSET_DEFAULT))
    parser.add_argument("--maskrcnn", default=None)
    parser.add_argument("--yolo", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default=str(ROOT / "work" / "models" / "notoverkil_eval"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-maskrcnn", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--maskrcnn-size", type=int, default=0, help="override maskrcnn input size (default 1536)")
    parser.add_argument("--rpn-post-nms", type=int, default=0, help="override RPN post-NMS top-n (memory saving at 1536)")
    parser.add_argument("--maskrcnn-detections", type=int, default=0, help="override detections_per_img")
    parser.add_argument("--data-root", default=None, help="competition data root (Kaggle mount or local)")
    parser.add_argument("--folds-json", default=None)
    parser.add_argument("--converted-dir", default=None, help="where to cache the converted yolo checkpoint")
    args = parser.parse_args(argv)
    _set_data_paths(args.data_root, args.folds_json)
    asset_dir = Path(args.asset_dir)
    maskrcnn_path = Path(args.maskrcnn) if args.maskrcnn else asset_dir / "p1-maskrcnn1536-long-f0" / "last.pt"
    yolo_path = Path(args.yolo) if args.yolo else asset_dir / "p1-yolo11m1536-long-f0" / "last.pt"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log(f"assets: maskrcnn={maskrcnn_path} yolo={yolo_path}")

    if args.mode == "diagnose":
        info = {"maskrcnn": diagnose_maskrcnn(maskrcnn_path), "yolo": diagnose_yolo(yolo_path)}
        (out_dir / "diagnose.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        return 0

    stems = fold0_val_stems()
    if args.limit:
        stems = stems[: args.limit]
    image_paths = resolve_image_paths(stems)
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[:3])
    log(f"fold{args.fold} val: {len(stems)} physical images")

    maskrcnn_state, _meta = load_maskrcnn_state(maskrcnn_path)
    conv1 = maskrcnn_state.get("backbone.body.conv1.weight")
    cls = maskrcnn_state.get("roi_heads.box_predictor.cls_score.weight")
    num_classes = int(cls.shape[0]) if cls is not None else 2
    in_channels = int(conv1.shape[1]) if conv1 is not None else 3
    log(f"maskrcnn: num_classes={num_classes} in_channels={in_channels}")
    maskrcnn = build_maskrcnn(num_classes, in_channels, MASK_RCNN_CFG["image_size"])
    missing_keys, unexpected_keys = maskrcnn.load_state_dict(maskrcnn_state, strict=False)
    log(f"maskrcnn load: missing={len(missing_keys)} unexpected={len(unexpected_keys)}")
    if missing_keys[:5]:
        log(f"  missing sample {missing_keys[:5]}")
    if unexpected_keys[:5]:
        log(f"  unexpected sample {unexpected_keys[:5]}")
    if args.rpn_post_nms:
        maskrcnn.rpn.post_nms_top_n_test = int(args.rpn_post_nms)
        maskrcnn.rpn.post_nms_top_n_train = int(args.rpn_post_nms)
        log(f"maskrcnn rpn.post_nms_top_n_test={args.rpn_post_nms}")
    if args.maskrcnn_detections:
        maskrcnn.roi_heads.detections_per_img = int(args.maskrcnn_detections)
        log(f"maskrcnn detections_per_img={args.maskrcnn_detections}")
    maskrcnn = maskrcnn.to(device)

    yolo_model = None
    if not args.skip_yolo:
        converted_dir = Path(args.converted_dir) if args.converted_dir else (yolo_path.parent if not yolo_path.parent.exists() else ROOT / "work" / "models" / "notoverkil_converted")
        yolo_model = load_yolo_checkpoint(yolo_path, converted_dir)

    gt_df = P.gt_df_from_coco(TRAIN_JSON, stems=stems)
    log(f"GT rows: {len(gt_df)}")

    results = []
    maskrcnn_preds = {stem: [] for stem in stems}
    yolo_preds = {stem: [] for stem in stems}
    if not args.skip_maskrcnn:
        started = time.time()
        maskrcnn_preds = infer_maskrcnn(maskrcnn, device, image_paths, size_override=args.maskrcnn_size or None)
        log(f"maskrcnn inference {time.time()-started:.0f}s")
        del maskrcnn
        torch.cuda.empty_cache()
        results.append(evaluate(maskrcnn_preds, f"maskrcnn{args.maskrcnn_size or MASK_RCNN_CFG['image_size']}_f0", gt_df, stems))
    if not args.skip_yolo and yolo_model is not None:
        started = time.time()
        yolo_preds = infer_yolo(yolo_model, device, image_paths)
        log(f"yolo inference {time.time()-started:.0f}s")
        results.append(evaluate(yolo_preds, "yolo11m1536_f0", gt_df, stems))
    if results and not args.skip_maskrcnn and not args.skip_yolo:
        fused = fuse(maskrcnn_preds, yolo_preds, MASK_RCNN_CFG["minimum_area"], MASK_RCNN_CFG["maximum_area"], max(MASK_RCNN_CFG["maximum_detections"], YOLO_CFG["maximum_detections"]))
        results.append(evaluate(fused, "fused_mask_level", gt_df, stems))
        to_pred_df({stem: fused.get(stem, []) for stem in stems}).to_csv(out_dir / "fused_pred.csv", index=False)

    # save prediction CSVs (small) for later fusion work
    if not args.skip_maskrcnn:
        to_pred_df({stem: maskrcnn_preds.get(stem, []) for stem in stems}).to_csv(out_dir / "maskrcnn_pred.csv", index=False)
    if not args.skip_yolo:
        to_pred_df({stem: yolo_preds.get(stem, []) for stem in stems}).to_csv(out_dir / "yolo_pred.csv", index=False)
    payload = {"fold": args.fold, "stems": len(stems), "configs": {"maskrcnn": MASK_RCNN_CFG, "yolo": YOLO_CFG}, "results": results}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log("BEST: " + json.dumps(max(results, key=lambda row: row["pq"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
