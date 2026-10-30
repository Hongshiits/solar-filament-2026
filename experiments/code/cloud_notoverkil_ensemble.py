"""Reproduce notoverkil's full ensemble pipeline on Kaggle T4 (cloud only).

Pipeline: 25 public weights (5 families x 5 folds)
  -> per-model 8-view dihedral TTA -> instance masks (2048) + scores
  -> consensus fusion over every instance of every model/view:
       greedy IoU>=0.5 clustering -> score-weighted pixel vote (>=0.5)
       -> area>=150 -> max_det 150 -> greedy de-overlap -> RLE
  -> official pooled PQ (fold0 honest OOF) or submission.csv (test).

Modes
-----
fold0-eval : fold0 models only (one per family) + consensus, official PQ on the
             harness fold0 val split (142 physical images).
test       : all models in --families -> consensus -> submission.csv for 180 test images.
assemble   : skip inference, reuse --cache-dirs to run consensus + PQ/submission.

Usage
-----
python cloud_notoverkil_ensemble.py --mode fold0-eval --families all
python cloud_notoverkil_ensemble.py --mode fold0-eval --families yolo11m-1536,yolo26m-1536
python cloud_notoverkil_ensemble.py --mode test --families yolo11m-1536,yolo26m-1536
python cloud_notoverkil_ensemble.py --mode test --shard-index 0 --shard-count 2
python cloud_notoverkil_ensemble.py --mode assemble --cache-dirs /kaggle/working/a /kaggle/working/b --submit

All products under /kaggle/working/solar_ensemble/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

H = W = 2048
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/solar_ensemble")
DIHEDRAL = [(0, False), (0, True), (1, False), (1, True), (2, False), (2, True), (3, False), (3, True)]
DEFAULT_FAMILIES = ["yolo11m-1536", "yolo11m-2048", "yolo26m-1536", "dense-convnext-small", "maskrcnn-1536"]
FAMILY_CFG = {
    "yolo11m-1536": {"imgsz": 1536, "score": 0.2, "nms": 0.7, "min_area": 150, "max_area": 50000, "max_det": 60},
    "yolo11m-2048": {"imgsz": 2048, "score": 0.2, "nms": 0.7, "min_area": 150, "max_area": 50000, "max_det": 60},
    "yolo26m-1536": {"imgsz": 1536, "score": 0.2, "nms": 0.7, "min_area": 150, "max_area": 50000, "max_det": 60},
    "dense-convnext-small": {"imgsz": 1024, "score": 0.3, "min_area": 150, "max_area": 50000, "max_det": 100},
    "maskrcnn-1536": {"imgsz": 1536, "score": 0.3, "mask_thr": 0.5, "min_area": 150, "max_area": 50000, "max_det": 100},
}
CONSENSUS = {"cluster_iou": 0.5, "pixel_vote": 0.5, "min_area": 150, "max_det": 150}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def safe_family(name: str) -> str:
    for key in FAMILY_CFG:
        if key in name:
            return key
    lowered = name.lower()
    if "yolo26" in lowered:
        return "yolo26m-1536"
    if "yolo11m2048" in lowered or "yolo11m-2048" in lowered:
        return "yolo11m-2048"
    if "yolo11m" in lowered:
        return "yolo11m-1536"
    if "dense" in lowered or "convnext" in lowered:
        return "dense-convnext-small"
    if "maskrcnn" in lowered:
        return "maskrcnn-1536"
    return "unknown"


# ----------------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------------


def ensure_packages() -> None:
    # Install optional packages that are missing from the Kaggle image (internet is enabled).
    import importlib
    import subprocess

    needed = []
    for module, package in (
        ("segmentation_models_pytorch", "segmentation-models-pytorch"),
        ("timm", "timm"),
    ):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            needed.append(package)
    if needed:
        print("[bootstrap] pip install " + " ".join(needed), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-input", *needed])


def find_competition() -> dict:
    candidates = sorted(INPUT.glob("**/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not candidates:
        raise FileNotFoundError("competition annotation json not found under /kaggle/input")
    root = candidates[0].parent.parent
    return {"root": root, "train_images": root / "train" / "train_images", "test_images": root / "test" / "test_images", "json": candidates[0]}


def find_folds(path: str | None = None) -> dict:
    if path:
        return json.load(open(path, encoding="utf-8"))
    for pattern in ("**/work/cache/folds.json", "**/folds.json"):
        candidates = sorted(INPUT.glob(pattern))
        if candidates:
            return json.load(open(candidates[0], encoding="utf-8"))
    raise FileNotFoundError("folds.json not found under /kaggle/input")


def find_manifest() -> tuple[dict | None, Path | None]:
    for path in sorted(INPUT.glob("**/manifest.json")):
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(payload, dict) and "models" in payload:
            return payload, path.parent
    return None, None


def discover_models(manifest: dict | None, base: Path | None) -> list[dict]:
    models = []
    if manifest and base is not None:
        for entry in manifest["models"]:
            run = entry["run"]
            weight = base / run / "last.pt"
            if not weight.exists():
                found = sorted(INPUT.glob(f"**/{run}/last.pt"))
                weight = found[0] if found else None
            if weight is None:
                continue
            config = weight.parent / "config.yaml"
            models.append(
                {
                    "run": run,
                    "family": entry.get("family", safe_family(run)),
                    "fold": entry.get("fold"),
                    "weight": weight,
                    "config": config if config.exists() else None,
                }
            )
    if not models:
        for weight in sorted(INPUT.glob("**/last.pt")):
            run = weight.parent.name
            match = re.search(r"f(\d)", run)
            models.append(
                {
                    "run": run,
                    "family": safe_family(run),
                    "fold": int(match.group(1)) if match else None,
                    "weight": weight,
                    "config": weight.parent / "config.yaml" if (weight.parent / "config.yaml").exists() else None,
                }
            )
    return models


# ----------------------------------------------------------------------------
# dihedral TTA
# ----------------------------------------------------------------------------
def dihedral_apply(image: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(image, k) if k else image
    if flip:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def dihedral_inverse_mask(mask: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = mask
    if flip:
        out = np.fliplr(out)
    if k:
        out = np.rot90(out, -k)
    return np.ascontiguousarray(out)


# ----------------------------------------------------------------------------
# instance cache (per model npz: stems / rles / scores)
# ----------------------------------------------------------------------------
def encode_instances(instances: list[dict]) -> tuple[list[str], list[float]]:
    from pycocotools import mask as mask_utils

    rles, scores = [], []
    for inst in instances:
        counts = mask_utils.encode(np.asfortranarray(inst["mask"].astype(np.uint8)))["counts"]
        rles.append(counts.decode("ascii") if isinstance(counts, bytes) else str(counts))
        scores.append(float(inst["score"]))
    return rles, scores


def decode_rle(text: str) -> np.ndarray:
    from pycocotools import mask as mask_utils

    counts = text.encode("ascii") if isinstance(text, str) else text
    return mask_utils.decode({"size": [H, W], "counts": counts}).astype(bool)


# ----------------------------------------------------------------------------
# model loaders / predictors
# ----------------------------------------------------------------------------
def load_yolo_any(weight: Path, converted_dir: Path):
    from ultralytics import YOLO

    try:
        return YOLO(str(weight))
    except Exception as error:  # noqa: BLE001
        log(f"  yolo direct load failed ({str(error)[:120]}); rebuilding from architecture")
    import torch
    from ultralytics.nn.tasks import SegmentationModel

    payload = torch.load(str(weight), map_location="cpu", weights_only=False)
    if not (isinstance(payload, dict) and "architecture" in payload and "ema" in payload):
        raise RuntimeError(f"unsupported yolo checkpoint {weight}")
    architecture = dict(payload["architecture"])
    architecture["task"] = "segment"
    model = SegmentationModel(architecture, ch=int(architecture.get("ch", 3)), nc=int(architecture.get("nc", 1)), verbose=False)
    missing, _unexpected = model.load_state_dict(payload["ema"], strict=False)
    if missing:
        raise RuntimeError(f"yolo rebuild missing keys: {missing[:5]}")
    model.task = "segment"
    model.names = {0: "filament"}
    model.args = {"task": "segment"}
    model.eval()
    converted_dir.mkdir(parents=True, exist_ok=True)
    target = converted_dir / f"{weight.parent.name}_converted.pt"
    torch.save({"model": model, "epoch": 0, "version": "8.4.191", "date": time.strftime("%Y-%m-%d"), "task": "segment", "train_args": {"task": "segment"}}, target)
    log(f"  yolo rebuilt -> {target.name}")
    del model
    torch.cuda.empty_cache()
    return YOLO(str(target))


def predict_yolo(model, image: np.ndarray, cfg: dict, device: str) -> list[dict]:
    result = model.predict(
        source=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        imgsz=cfg["imgsz"], conf=cfg["score"], iou=cfg["nms"], agnostic_nms=True,
        retina_masks=True, max_det=cfg["max_det"], verbose=False, device=device,
    )[0]
    if result.masks is None or result.boxes is None or len(result.masks.data) == 0:
        return []
    masks = result.masks.data.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy().astype(float)
    instances = []
    for score, mask in zip(scores, masks):
        mask = mask.astype(np.uint8)
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        area = int(mask.sum())
        if area < cfg["min_area"] or area > cfg["max_area"]:
            continue
        instances.append({"mask": mask.astype(bool), "score": float(score)})
    return instances


def load_maskrcnn(weight: Path):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    payload = torch.load(str(weight), map_location="cpu", weights_only=False)
    state = payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "ema"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    cls = conv1 = None
    for key, value in state.items():
        if key.endswith("box_predictor.cls_score.weight"):
            cls = value
        if key.endswith("backbone.body.conv1.weight"):
            conv1 = value
    num_classes = int(cls.shape[0]) if cls is not None else 2
    in_channels = int(conv1.shape[1]) if conv1 is not None else 3
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=1536, max_size=1536)
    if in_channels != 3:
        old = model.backbone.body.conv1
        model.backbone.body.conv1 = torch.nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=old.bias is not None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    in_feats_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feats_mask, 256, num_classes)
    model.load_state_dict(state, strict=False)
    return model


def predict_maskrcnn(model, image: np.ndarray, cfg: dict, device: str) -> list[dict]:
    import torch

    size = cfg["imgsz"]
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().unsqueeze(0).repeat(3, 1, 1) / 255.0
    with torch.no_grad():
        out = model([tensor.to(device)])[0]
    scores = out["scores"].detach().cpu().numpy().astype(float)
    masks = out["masks"].detach().cpu().numpy()[:, 0]
    instances = []
    for score, mask in zip(scores, masks):
        if score < cfg["score"]:
            continue
        mask_full = cv2.resize((mask > cfg["mask_thr"]).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(mask_full.sum())
        if area < cfg["min_area"] or area > cfg["max_area"]:
            continue
        instances.append({"mask": mask_full, "score": float(score)})
    return instances


def load_dense_model(weight: Path):
    import torch
    import segmentation_models_pytorch as smp

    payload = torch.load(str(weight), map_location="cpu", weights_only=False)
    state = payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "ema"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    log(f"  dense keys sample: {list(state.keys())[:6]}")
    for encoder in ("tu-convnext_small", "tu-convnext_tiny", "convnext_small", "resnet34", "resnet18"):
        try:
            model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=1, classes=1)
            missing, _unexpected = model.load_state_dict(state, strict=False)
            if len(missing) == 0:
                log(f"  dense: loaded as smp.Unet encoder={encoder}")
                return {"kind": "smp", "model": model, "encoder": encoder}
        except Exception:  # noqa: BLE001
            continue
    return None


def predict_dense(model, image: np.ndarray, cfg: dict, device: str) -> list[dict]:
    import torch
    import segmentation_models_pytorch as smp

    size = cfg["imgsz"]
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean, std = 0.5, 0.25
    tensor = torch.from_numpy((resized - mean) / std).float()[None, None].to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob_full = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    import cv2 as _cv2

    seed = (prob_full >= cfg["score"]).astype(np.uint8)
    n, labels, stats, _cent = _cv2.connectedComponentsWithStats(seed, connectivity=8)
    instances = []
    for comp in range(1, n):
        area = int(stats[comp, _cv2.CC_STAT_AREA])
        if area < cfg["min_area"] or area > cfg["max_area"]:
            continue
        mask = labels == comp
        score = float(prob_full[mask].mean())
        instances.append({"mask": mask, "score": score})
    return instances


# ----------------------------------------------------------------------------
# inference driver (model-outer, cache instances per model as npz)
# ----------------------------------------------------------------------------
def run_inference(models: list[dict], image_paths: list[Path], stems: list[str], cache_dir: Path, device: str, views: int, skip_maskrcnn: bool, skip_convnext: bool) -> list[dict]:
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, entry in enumerate(models):
        family = entry["family"]
        if skip_maskrcnn and family.startswith("maskrcnn"):
            continue
        if skip_convnext and family.startswith("dense"):
            continue
        cfg = dict(FAMILY_CFG.get(family, FAMILY_CFG["yolo11m-1536"]))
        started = time.time()
        log(f"[{index + 1}/{len(models)}] {entry['run']} family={family} fold={entry['fold']} imgsz={cfg['imgsz']}")
        kind = "yolo"
        model = None
        try:
            if family.startswith("maskrcnn"):
                kind = "maskrcnn"
                model = load_maskrcnn(entry["weight"]).to(device).eval()
            elif family.startswith("dense"):
                loaded = load_dense_model(entry["weight"])
                if loaded is None:
                    log("  skipped (unknown dense architecture)")
                    continue
                kind = "dense"
                model = loaded["model"].to(device).eval()
            else:
                model = load_yolo_any(entry["weight"], WORK / "converted")
        except Exception as error:  # noqa: BLE001
            log(f"  FAILED to load: {str(error)[:300]}")
            continue

        all_stems, all_rles, all_scores = [], [], []
        for number, (stem, image_path) in enumerate(zip(stems, image_paths), 1):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            if image.shape != (H, W):
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
            instances = []
            for k, flip in DIHEDRAL[: max(1, min(views, 8))]:
                view = dihedral_apply(image, k, flip)
                if kind == "yolo":
                    view_instances = predict_yolo(model, view, cfg, device)
                elif kind == "maskrcnn":
                    view_instances = predict_maskrcnn(model, view, cfg, device)
                else:
                    view_instances = predict_dense(model, view, cfg, device)
                for inst in view_instances:
                    inst["mask"] = dihedral_inverse_mask(inst["mask"], k, flip)
                instances.extend(view_instances)
            rles, scores = encode_instances(instances)
            all_stems.extend([stem] * len(rles))
            all_rles.extend(rles)
            all_scores.extend(scores)
            if number % 20 == 0:
                log(f"    {number}/{len(stems)} images ({time.time() - started:.0f}s)")
        np.savez_compressed(cache_dir / f"model_{index:03d}.npz", stems=np.asarray(all_stems), rles=np.asarray(all_rles, dtype=object), scores=np.asarray(all_scores, dtype=np.float32))
        log(f"  cached {len(all_stems)} instances -> model_{index:03d}.npz; {(time.time() - started) / 60:.1f} min")
        del model
        torch.cuda.empty_cache()
        summaries.append({"run": entry["run"], "family": family, "fold": entry["fold"], "kind": kind, "instances": len(all_stems)})
    return summaries


# ----------------------------------------------------------------------------
# consensus fusion
# ----------------------------------------------------------------------------
def make_entry(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    return {"x0": x0, "y0": y0, "m": mask[y0 : int(ys.max()) + 1, x0 : int(xs.max()) + 1].astype(np.uint8)}


def entry_area(entry) -> int:
    return int(entry["m"].sum())


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


def entry_to_mask(entry) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[entry["y0"] : entry["y0"] + entry["m"].shape[0], entry["x0"] : entry["x0"] + entry["m"].shape[1]] = entry["m"] > 0
    return mask


def consensus_for_image(instances: list[dict], cfg: dict = CONSENSUS) -> list[dict]:
    """Greedy IoU clustering + score-weighted pixel vote -> final disjoint masks.

    Pixel voting is done inside the cluster's union bounding box (not the full
    2048x2048 image): with ~80 instances/image in stage 1 and ~4000 in stage 2
    this is 50-500x faster than full-image float accumulators.
    """
    if not instances:
        return []
    prepared = []
    for inst in instances:
        mask = inst["mask"]
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        entry = {"x0": x0, "y0": y0, "m": mask[y0 : int(ys.max()) + 1, x0 : int(xs.max()) + 1].astype(np.uint8)}
        prepared.append({"entry": entry, "score": float(inst["score"])})
    prepared.sort(key=lambda item: -item["score"])
    clusters: list[dict] = []
    for item in prepared:
        for cluster in clusters:
            if entry_iou(item["entry"], cluster["representative"]) >= cfg["cluster_iou"]:
                cluster["members"].append(item)
                break
        else:
            clusters.append({"representative": item["entry"], "members": [item]})
    results = []
    for cluster in clusters:
        members = cluster["members"]
        x0 = min(member["entry"]["x0"] for member in members)
        y0 = min(member["entry"]["y0"] for member in members)
        x1 = max(member["entry"]["x0"] + member["entry"]["m"].shape[1] for member in members)
        y1 = max(member["entry"]["y0"] + member["entry"]["m"].shape[0] for member in members)
        height, width = y1 - y0, x1 - x0
        accumulator = np.zeros((height, width), dtype=np.float32)
        weight_sum = 0.0
        for member in members:
            entry = member["entry"]
            local = np.zeros((height, width), dtype=np.float32)
            local[
                entry["y0"] - y0 : entry["y0"] - y0 + entry["m"].shape[0],
                entry["x0"] - x0 : entry["x0"] - x0 + entry["m"].shape[1],
            ] = entry["m"]
            accumulator += local * member["score"]
            weight_sum += member["score"]
        if weight_sum <= 0:
            continue
        vote = accumulator / weight_sum
        final = vote >= cfg["pixel_vote"]
        area = int(final.sum())
        if area < cfg["min_area"]:
            continue
        ys, xs = np.where(final)
        if len(xs) == 0:
            continue
        ex0, ey0 = int(xs.min()), int(ys.min())
        ex1, ey1 = int(xs.max()) + 1, int(ys.max()) + 1
        entry = {"x0": x0 + ex0, "y0": y0 + ey0, "m": final[ey0:ey1, ex0:ex1].astype(np.uint8)}
        score = float(np.mean([member["score"] for member in members]))
        results.append({"entry": entry, "score": score})
    results.sort(key=lambda item: -item["score"])
    results = results[: cfg["max_det"]]
    kept_entries: list[dict] = []
    kept_scores: list[float] = []
    for item in results:
        updated = item["entry"]
        for previous in kept_entries:
            updated = subtract_entry(updated, previous)
            if updated is None:
                break
        if updated is None or entry_area(updated) < cfg["min_area"]:
            continue
        kept_entries.append(updated)
        kept_scores.append(item["score"])
    return [{"mask": entry_to_mask(entry), "score": score} for entry, score in zip(kept_entries, kept_scores)]


def consensus_all(cache_dirs: list[Path], stems: list[str], families_filter: set[str] | None = None, cache_meta: dict[str, dict] | None = None) -> dict[str, list[dict]]:
    """Load caches (stem->instances), optionally restrict to families, run consensus per image."""
    from pycocotools import mask as mask_utils

    # index metadata only first (stems + offsets), decode lazily per image
    per_cache: list[dict] = []
    for cache_dir in cache_dirs:
        for path in sorted(cache_dir.glob("model_*.npz")):
            payload = np.load(path, allow_pickle=True)
            per_cache.append(
                {
                    "path": path,
                    "stems": payload["stems"].tolist(),
                    "rles": payload["rles"].tolist(),
                    "scores": payload["scores"].tolist(),
                    "family": (cache_meta or {}).get(path.stem, {}).get("family"),
                }
            )
    by_stem: dict[str, list[dict]] = {stem: [] for stem in stems}
    for cache in per_cache:
        if families_filter is not None and cache.get("family") is not None and cache["family"] not in families_filter:
            continue
        for stem, rle, score in zip(cache["stems"], cache["rles"], cache["scores"]):
            if stem not in by_stem:
                by_stem[stem] = []
            by_stem[stem].append({"rle": rle, "score": float(score)})
    output: dict[str, list[dict]] = {}
    started = time.time()
    for number, stem in enumerate(stems, 1):
        instances = []
        for item in by_stem.get(stem, []):
            instances.append({"mask": decode_rle(item["rle"]), "score": item["score"]})
        output[stem] = consensus_for_image(instances)
        if number % 20 == 0:
            log(f"consensus {number}/{len(stems)} images ({time.time() - started:.0f}s)")
    return output


# ----------------------------------------------------------------------------
# official PQ + submission
# ----------------------------------------------------------------------------
def load_pq_metric():
    candidates = [Path(__file__).resolve().parent / "pq_metric.py"]
    for pattern in ("**/pq_metric.py", "**/src/pq_metric.py"):
        candidates.extend(sorted(INPUT.glob(pattern)))
    for path in candidates:
        if not path.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("pq_metric_cloud", path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            if hasattr(module, "gt_df_from_coco") and hasattr(module, "evaluate"):
                log(f"pq_metric: {path}")
                return module
        except Exception:  # noqa: BLE001
            continue
    return None


def embedded_pq(gt_json: Path, pred_by_stem: dict[str, list[dict]], stems: list[str]) -> dict:
    """Official pooled PQ: TP = every pair with IoU>0.5; FP/FN = unmatched; pooled over annotator records."""
    from pycocotools import mask as mask_utils

    coco = json.load(open(gt_json, encoding="utf-8"))
    images = {str(image["id"]): image for image in coco["images"]}
    ann_by_image: dict[str, list[dict]] = {}
    for annotation in coco["annotations"]:
        ann_by_image.setdefault(str(annotation["image_id"]), []).append(annotation)
    ids_by_stem: dict[str, list[str]] = {}
    for image_id, image in images.items():
        stem = image_id.split("-", 1)[-1]
        ids_by_stem.setdefault(stem, []).append(image_id)
    tp = 0
    sum_iou = 0.0
    fp = 0
    fn = 0
    for stem in stems:
        pred = [mask_utils.encode(np.asfortranarray(item["mask"].astype(np.uint8))) for item in pred_by_stem.get(stem, [])]
        for image_id in sorted(ids_by_stem.get(stem, [])):
            gts = []
            for annotation in ann_by_image.get(image_id, []):
                if annotation.get("iscrowd", 0):
                    continue
                segmentation = annotation.get("segmentation")
                if isinstance(segmentation, list):
                    rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, H, W))
                else:
                    rle = segmentation
                gts.append(rle)
            if not gts and not pred:
                continue
            if not pred:
                fn += len(gts)
                continue
            if not gts:
                fp += len(pred)
                continue
            iou = np.asarray(mask_utils.iou(gts, pred, [0] * len(pred)), dtype=np.float64)
            hit = iou > 0.5
            tp += int(hit.sum())
            sum_iou += float(iou[hit].sum())
            fp += int((hit.sum(axis=0) == 0).sum())
            fn += int((hit.sum(axis=1) == 0).sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": sum_iou / denominator if denominator else 0.0,
        "sq": sum_iou / tp if tp else 0.0,
        "rq": tp / denominator if denominator else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sum_iou": sum_iou,
    }


def evaluate_consensus(pred: dict[str, list[dict]], stems: list[str], gt_json: Path, label: str) -> dict:
    metric = load_pq_metric()
    if not pred or not any(pred.get(stem) for stem in stems):
        log(f"PQ {label}: EMPTY predictions (skipped family) -> pq=0.0")
        return {"label": label, "pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": 0}

    if metric is not None:
        rows = []
        for stem in stems:
            for index, item in enumerate(pred.get(stem, []), 1):
                rle = item["mask"]
                from pycocotools import mask as mask_utils

                counts = mask_utils.encode(np.asfortranarray(rle.astype(np.uint8)))["counts"]
                rows.append({"filament_id": f"{stem}_{index}", "segmentation_rle": counts.decode("ascii") if isinstance(counts, bytes) else str(counts)})
        import pandas as pd

        gt_df = metric.gt_df_from_coco(gt_json, stems=stems)
        result = metric.evaluate(gt_df, pd.DataFrame(rows))
        report = {key: value for key, value in result.items() if key not in ("per_image", "tp_ious")}
    else:
        report = embedded_pq(gt_json, pred, stems)
    log(f"PQ {label}: pq={report['pq']:.6f} sq={report['sq']:.6f} rq={report['rq']:.6f} tp={report['tp']} fp={report['fp']} fn={report['fn']}")
    return {"label": label, **{key: report[key] for key in ("pq", "sq", "rq", "tp", "fp", "fn")}}


def write_submission(pred: dict[str, list[dict]], path: Path) -> int:
    from pycocotools import mask as mask_utils

    rows = []
    for stem in sorted(pred):
        for index, item in enumerate(pred[stem], 1):
            counts = mask_utils.encode(np.asfortranarray(item["mask"].astype(np.uint8)))["counts"]
            rows.append({"filament_id": f"{stem}_{index}", "segmentation_rle": counts.decode("ascii") if isinstance(counts, bytes) else str(counts)})
    import pandas as pd

    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    n_images = int(df["filament_id"].str.rsplit("_", n=1).str[0].nunique()) if len(df) else 0
    log(f"submission: {len(df)} instances / {n_images} images -> {path}")
    return len(df)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="fold0-eval", choices=["fold0-eval", "test", "assemble"])
    parser.add_argument("--families", default="all", help="all or comma separated family names")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--skip-maskrcnn", action="store_true")
    parser.add_argument("--skip-convnext", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--cache-dirs", nargs="*", default=None)
    parser.add_argument("--out-dir", default=str(WORK))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--folds-json", default=None)
    parser.add_argument("--submit", action="store_true", help="write submission.csv in assemble mode")
    parser.add_argument("--no-maskrcnn-eval", action="store_true", help="exclude maskrcnn from consensus")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_root:
        root = Path(args.data_root)
        competition = {
            "root": root,
            "train_images": root / "train" / "train_images",
            "test_images": root / "test" / "test_images",
            "json": root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        }
    else:
        ensure_packages()
    competition = find_competition()
    families = DEFAULT_FAMILIES if args.families == "all" else [item.strip() for item in args.families.split(",") if item.strip()]
    log(f"mode={args.mode} families={families} views={args.views}")

    if args.mode == "assemble":
        if not args.cache_dirs:
            raise ValueError("--cache-dirs required for assemble mode")
        folds = find_folds(args.folds_json)
        if args.submit:
            stems = sorted(path.stem for path in competition["test_images"].glob("*.jpeg"))
            if args.limit:
                stems = stems[: args.limit]
            log(f"assemble test: {len(stems)} stems")
            pred = consensus_all([Path(p) for p in args.cache_dirs], stems)
            rows = write_submission(pred, out_dir / "submission.csv")
            meta = {"mode": "assemble-test", "stems": len(stems), "rows": rows, "families": families, "views": args.views}
        else:
            stems = sorted(folds["folds"][str(args.fold)])
            if args.limit:
                stems = stems[: args.limit]
            pred = consensus_all([Path(p) for p in args.cache_dirs], stems)
            report = evaluate_consensus(pred, stems, competition["json"], "assemble-fold")
            meta = {"mode": "assemble-eval", "stems": len(stems), "result": report, "families": families, "views": args.views}
        (out_dir / "assemble_results.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return 0

    # inference modes need model list
    manifest, manifest_base = find_manifest()
    models = discover_models(manifest, manifest_base)
    log(f"discovered {len(models)} model checkpoints")
    if args.mode == "fold0-eval":
        models = [entry for entry in models if entry.get("fold") == args.fold]
        if not models:
            raise RuntimeError("no fold0 model checkpoints found")
        models = [entry for entry in models if entry["family"] in families]
        # one model per family for the honest fold0 stage
        selected = {}
        for entry in models:
            selected.setdefault(entry["family"], entry)
        models = list(selected.values())
        stems = sorted(find_folds(args.folds_json)["folds"][str(args.fold)])
        image_paths = [competition["train_images"] / f"{stem}.jpeg" for stem in stems]
        split = "fold0val"
    else:
        models = [entry for entry in models if entry["family"] in families]
        stems = sorted(path.stem for path in competition["test_images"].glob("*.jpeg"))
        image_paths = [competition["test_images"] / f"{stem}.jpeg" for stem in stems]
        split = "test"
    if args.shard_count > 1:
        models = [entry for index, entry in enumerate(models) if index % args.shard_count == args.shard_index]
    if args.limit:
        stems = stems[: args.limit]
        image_paths = image_paths[: args.limit]
    log(f"{split}: {len(stems)} images, {len(models)} models")

    cache_dir = out_dir / f"cache_{split}_shard{args.shard_index}"
    summaries = run_inference(models, image_paths, stems, cache_dir, args.device, args.views, args.skip_maskrcnn, args.skip_convnext)
    (out_dir / f"inference_summary_{split}_shard{args.shard_index}.json").write_text(json.dumps(summaries, indent=1), encoding="utf-8")

    if args.mode == "fold0-eval":
        cache_meta = {f"model_{index:03d}": {"family": entry["family"], "fold": entry["fold"], "run": entry["run"]} for index, entry in enumerate(models)}
        results = []
        # per-family consensus (8 views within one family)
        for family in sorted({entry["family"] for entry in models}):
            pred = consensus_family(cache_dir, family, stems, cache_meta)
            if not any(pred.get(stem) for stem in stems):
                log(f"skip family {family}: empty cache")
                continue
            results.append(evaluate_consensus(pred, stems, competition["json"], f"consensus_{family}"))
        # cross-family consensus (maskrcnn excluded by default: known weak / spec mismatch)
        include_maskrcnn = not args.skip_maskrcnn and not args.no_maskrcnn_eval
        pred_by_family = {}
        if include_maskrcnn:
            pred_all = consensus_family(cache_dir, None, stems, cache_meta)
            results.append(evaluate_consensus(pred_all, stems, competition["json"], "consensus_all_families"))
        else:
            families_kept = [family for family in sorted({entry["family"] for entry in models}) if not family.startswith("maskrcnn")]
            pred_parts = [consensus_family(cache_dir, family, stems, cache_meta) for family in families_kept]
            pred_all = {}
            for stem in stems:
                merged = []
                for part in pred_parts:
                    merged.extend(part.get(stem, []))
                pred_all[stem] = consensus_for_image([{"mask": item["mask"], "score": item["score"]} for item in merged])
            results.append(evaluate_consensus(pred_all, stems, competition["json"], "consensus_all_families"))
        (out_dir / "fold0_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        write_submission(pred_all, out_dir / "fold0_consensus_pred.csv")
        log("FOLD0 RESULTS: " + json.dumps(results))
    else:
        if args.shard_count > 1:
            log("shard done; run assemble with --cache-dirs of all shards")
        else:
            pred = consensus_all([cache_dir], stems)
            write_submission(pred, out_dir / "submission.csv")
    return 0


def consensus_family(cache_dir: Path, family: str | None, stems: list[str], cache_meta: dict[str, dict]) -> dict[str, list[dict]]:
    """Consensus restricted to one family (or all when family is None)."""
    from pycocotools import mask as mask_utils

    by_stem: dict[str, list[dict]] = {stem: [] for stem in stems}
    for path in sorted(cache_dir.glob("model_*.npz")):
        meta = cache_meta.get(path.stem, {})
        if family is not None and meta.get("family") != family:
            continue
        payload = np.load(path, allow_pickle=True)
        for stem, rle, score in zip(payload["stems"].tolist(), payload["rles"].tolist(), payload["scores"].tolist()):
            if stem in by_stem:
                by_stem[stem].append({"rle": rle, "score": float(score)})
    output = {}
    for stem in stems:
        instances = [{"mask": decode_rle(item["rle"]), "score": item["score"]} for item in by_stem[stem]]
        output[stem] = consensus_for_image(instances)
    return output


if __name__ == "__main__":
    sys.exit(main())
