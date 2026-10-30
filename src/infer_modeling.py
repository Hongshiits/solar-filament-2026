"""Inference / PQ evaluation / submission for the self-trained YOLO11-seg route.

Pipeline (per physical image)
-----------------------------
1. yolo11*-seg runs at imgsz=1024 -> coarse instance masks + detector scores
   (one pass at a low candidate conf; every threshold is swept afterwards).
2. duplicate seeds removed (mask IoU > 0.6, keep higher score).
3. every seed gets a full-resolution 2048 square crop (context 1.8, min 96),
   resized to 256, 3 channels [gray, CLAHE, seed]; the ResNet18 U-Net refiner
   predicts a per-pixel filament probability map with 4-flip TTA + topology
   guard (accept TTA only when area / dice / component counts stay sane).
4. ALL intermediates (detector score, crop bounds, refiner probability maps,
   seed mask) are cached per image to work/models/preds/<split>.npz, so the
   threshold sweep is pure CPU and never re-runs the GPU.
5. PQ uses the OFFICIAL definition: TP = every GT/pred pair with IoU > 0.5 (no
   one-to-one constraint), FP/FN = unmatched predictions/GT, and
   PQ = sum(IoU_TP) / (TP + 0.5 FP + 0.5 FN), pooled over every annotator record
   of every evaluated physical image (harness src/pq_metric.py is used as the
   authoritative implementation when available).
6. Submission masks are greedily de-overlapped so that no two masks of one
   image share a pixel (official hard constraint), and per-instance confidence
   (detector_conf x refiner_conf) is exported for later model integration.

Commands
--------
python src/infer_modeling.py cache-preds --fold 0 --label-source all
python src/infer_modeling.py sweep       --fold 0 --label-source all
python src/infer_modeling.py evaluate    --fold 0 --label-source all --conf 0.20 --ref 0.65 --tta guard --morph none --min-area 50 --order combo
python src/infer_modeling.py cache-preds --test --label-source all
python src/infer_modeling.py predict-test --fold 0 --label-source all --conf 0.20 --ref 0.65
python src/infer_modeling.py oof --folds 0,1,2,3,4 --label-source all
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_yolo_seg import (  # noqa: E402
    MODELS,
    TEST_IMAGES,
    TRAIN_IMAGES,
    annotation_to_rle,
    build_indices,
    load_coco,
    load_grouped_folds,
    log,
    read_lock,
)
from train_refiner import CONTEXT, MIN_CROP, SeedRefiner, square_bounds_from_mask  # noqa: E402

IMAGE_SIZE = 2048
IMAGE_SIZE_DET = 1024
CAND_CONF = 0.05
NMS_IOU = 0.50
SEED_DUPLICATE_IOU = 0.60
MATCH_IOU = 0.50
PROB_CROP = 256
PREDS_ROOT = MODELS / "preds"

# threshold sweep grids (PQ-aware tuning, per Lead's checklist)
CONF_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
REF_GRID = [0.45, 0.55, 0.65, 0.75]
MORPH_GRID = ["none", "dilate1", "erode1"]
MIN_AREA_GRID = [20, 50, 96]
ORDER_GRID = ["combo"]
TTA_GRID = ["guard"]


# ----------------------------------------------------------------------------
# bbox-local mask entries (fast, exact integer IoU)
# ----------------------------------------------------------------------------


def entry_from_full(mask: np.ndarray) -> dict | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    return {"x0": x0, "y0": y0, "m": np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=np.uint8)}


def entry_from_rle(rle: dict) -> dict | None:
    return entry_from_full(mask_utils.decode(rle).astype(bool))


def entries_from_submask(submask: np.ndarray, x0: int, y0: int) -> dict | None:
    ys, xs = np.where(submask)
    if len(xs) == 0:
        return None
    bx0, by0 = int(xs.min()), int(ys.min())
    bx1, by1 = int(xs.max()) + 1, int(ys.max()) + 1
    return {"x0": x0 + bx0, "y0": y0 + by0, "m": np.ascontiguousarray(submask[by0:by1, bx0:bx1], dtype=np.uint8)}


def entry_area(entry: dict) -> int:
    return int(entry["m"].sum())


def entry_to_full(entry: dict) -> np.ndarray:
    full = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    full[entry["y0"] : entry["y0"] + entry["m"].shape[0], entry["x0"] : entry["x0"] + entry["m"].shape[1]] = entry["m"] > 0
    return full


def entries_iou(first: dict, second: dict) -> float:
    x0 = max(first["x0"], second["x0"])
    y0 = max(first["y0"], second["y0"])
    x1 = min(first["x0"] + first["m"].shape[1], second["x0"] + second["m"].shape[1])
    y1 = min(first["y0"] + first["m"].shape[0], second["y0"] + second["m"].shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    a = first["m"][y0 - first["y0"] : y1 - first["y0"], x0 - first["x0"] : x1 - first["x0"]]
    b = second["m"][y0 - second["y0"] : y1 - second["y0"], x0 - second["x0"] : x1 - second["x0"]]
    intersection = int(np.logical_and(a, b).sum())
    if intersection == 0:
        return 0.0
    union = entry_area(first) + entry_area(second) - intersection
    return float(intersection / union) if union > 0 else 0.0


def morph_entry(entry: dict, mode: str) -> dict | None:
    """Morphology (+-1 px at 2048 resolution) on the bbox-local mask."""
    if mode == "none":
        return entry
    padded = np.pad(entry["m"], 1)
    kernel = np.ones((3, 3), dtype=np.uint8)
    if mode == "dilate1":
        result = cv2.dilate(padded, kernel)
    elif mode == "erode1":
        result = cv2.erode(padded, kernel)
    else:
        raise ValueError(mode)
    ys, xs = np.where(result)
    if len(xs) == 0:
        return None
    bx0, by0 = int(xs.min()), int(ys.min())
    bx1, by1 = int(xs.max()) + 1, int(ys.max()) + 1
    return {
        "x0": entry["x0"] - 1 + bx0,
        "y0": entry["y0"] - 1 + by0,
        "m": np.ascontiguousarray(result[by0:by1, bx0:bx1], dtype=np.uint8),
    }


def subtract_entry(target: dict, other: dict) -> dict | None:
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
    return {
        "x0": target["x0"] + bx0,
        "y0": target["y0"] + by0,
        "m": np.ascontiguousarray(updated[by0:by1, bx0:bx1], dtype=np.uint8),
    }


def resolve_entries(entries: list[dict], scores: list[float], minimum_area: int):
    kept: list[dict] = []
    kept_scores: list[float] = []
    if not entries:
        return [], []
    for index in np.argsort(scores)[::-1]:
        entry = entries[int(index)]
        if entry_area(entry) < minimum_area:
            continue
        updated = entry
        for previous in kept:
            updated = subtract_entry(updated, previous)
            if updated is None:
                break
        if updated is None or entry_area(updated) < minimum_area:
            continue
        kept.append(updated)
        kept_scores.append(float(scores[int(index)]))
    return kept, kept_scores


# ----------------------------------------------------------------------------
# Official PQ
# ----------------------------------------------------------------------------


def pq_accumulate(gt_entry_sets: list[list[dict]], pred_entries: list[dict]) -> tuple[int, float, int, int]:
    tp = 0
    sum_iou = 0.0
    fp = 0
    fn = 0
    n_pred = len(pred_entries)
    for gt_entries in gt_entry_sets:
        n_gt = len(gt_entries)
        if n_gt == 0 and n_pred == 0:
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        if n_gt == 0:
            fp += n_pred
            continue
        matched_pred = [False] * n_pred
        matched_gt = [False] * n_gt
        for gt_index, gt_entry in enumerate(gt_entries):
            for pred_index, pred_entry in enumerate(pred_entries):
                iou = entries_iou(gt_entry, pred_entry)
                if iou > MATCH_IOU:
                    tp += 1
                    sum_iou += iou
                    matched_gt[gt_index] = True
                    matched_pred[pred_index] = True
        fp += matched_pred.count(False)
        fn += matched_gt.count(False)
    return tp, sum_iou, fp, fn


def finalize_pq(tp: int, sum_iou: float, fp: int, fn: int) -> dict:
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": sum_iou / denominator if denominator > 0 else 0.0,
        "sq": sum_iou / tp if tp > 0 else 0.0,
        "rq": tp / denominator if denominator > 0 else 0.0,
        "tp": tp,
        "sum_iou": sum_iou,
        "fp": fp,
        "fn": fn,
    }


# ----------------------------------------------------------------------------
# Model loading / detection / refinement
# ----------------------------------------------------------------------------


def load_refiner(checkpoint: Path, device: str) -> SeedRefiner:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = SeedRefiner(pretrained=False)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_detector(weights: Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


def detect_seeds(detector, image_bgr: np.ndarray, device: str, conf: float = CAND_CONF) -> list[dict]:
    result = detector.predict(
        source=image_bgr,
        imgsz=IMAGE_SIZE_DET,
        conf=conf,
        iou=NMS_IOU,
        agnostic_nms=True,
        retina_masks=False,
        verbose=False,
        device=device,
        max_det=300,
    )[0]
    if result.masks is None or result.boxes is None or len(result.masks.data) == 0:
        return []
    masks = result.masks.data.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy().astype(float)
    seeds = []
    for mask, score in zip(masks, scores):
        mask = mask.astype(np.uint8)
        if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        if int(mask.sum()) < 48:
            continue
        seeds.append({"mask": mask.astype(bool), "score": float(score)})
    return seeds


def dedupe_seeds(seeds: list[dict], threshold: float = SEED_DUPLICATE_IOU) -> list[dict]:
    for seed in seeds:
        seed["entry"] = entry_from_full(seed["mask"])
    kept: list[dict] = []
    for seed in sorted(seeds, key=lambda item: -item["score"]):
        if all(entries_iou(seed["entry"], other["entry"]) < threshold for other in kept):
            kept.append(seed)
    return kept


@torch.no_grad()
def refine_seeds(image: np.ndarray, seeds: list[dict], refiner: SeedRefiner, device: str, chunk: int = 8) -> list[dict]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    prepared = []
    for seed in seeds:
        x0, y0, x1, y1 = square_bounds_from_mask(seed["mask"], CONTEXT, MIN_CROP)
        image_crop = cv2.resize(image[y0:y1, x0:x1], (PROB_CROP, PROB_CROP), interpolation=cv2.INTER_LINEAR)
        seed_crop = cv2.resize(
            seed["mask"][y0:y1, x0:x1].astype(np.uint8), (PROB_CROP, PROB_CROP), interpolation=cv2.INTER_NEAREST
        )
        # NOTE: SeedRefiner.forward applies the ImageNet normalisation internally.
        features = np.stack([image_crop, clahe.apply(image_crop), seed_crop * 255], axis=0).astype(np.float32) / 255.0
        prepared.append(
            {
                "features": features,
                "bounds": (x0, y0, x1, y1),
                "score": seed["score"],
                "seed_crop": seed_crop.astype(bool),
            }
        )
    for start in range(0, len(prepared), chunk):
        batch_items = prepared[start : start + chunk]
        batch = torch.from_numpy(np.stack([item["features"] for item in batch_items])).to(device)
        base = torch.sigmoid(refiner(batch))
        accumulator = base.clone()
        for dims in ((-1,), (-2,), (-2, -1)):
            accumulator += torch.flip(torch.sigmoid(refiner(torch.flip(batch, dims=dims))), dims=dims)
        base_np = base[:, 0].detach().cpu().numpy().astype(np.float32)
        tta_np = (accumulator / 4.0)[:, 0].detach().cpu().numpy().astype(np.float32)
        for offset, item in enumerate(batch_items):
            item["prob_base"] = base_np[offset]
            item["prob_tta"] = tta_np[offset]
    return prepared


def predict_image(image_path: Path, detector, refiner, device: str) -> list[dict]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    seeds = dedupe_seeds(detect_seeds(detector, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), device))
    if not seeds:
        return []
    return refine_seeds(image, seeds, refiner, device)


# ----------------------------------------------------------------------------
# Config -> entries
# ----------------------------------------------------------------------------


def choose_probability(prob_base: np.ndarray, prob_tta: np.ndarray, threshold: float, tta_mode: str) -> np.ndarray:
    if tta_mode == "none":
        return prob_base
    base_mask = prob_base >= threshold
    candidate_mask = prob_tta >= threshold
    use_candidate = True
    if base_mask.any():
        area_ratio = candidate_mask.sum() / max(base_mask.sum(), 1)
        intersection = np.logical_and(base_mask, candidate_mask).sum()
        dice = float((2 * intersection + 1e-7) / (base_mask.sum() + candidate_mask.sum() + 1e-7))
        components_base = max(cv2.connectedComponents(base_mask.astype(np.uint8), connectivity=8)[0] - 1, 0)
        components_candidate = max(cv2.connectedComponents(candidate_mask.astype(np.uint8), connectivity=8)[0] - 1, 0)
        use_candidate = 0.60 <= area_ratio <= 1.70 and dice >= 0.62 and components_candidate <= components_base + 1
    return prob_tta if use_candidate else prob_base


def seed_entries_for(refined: list[dict], ref_threshold: float, tta_mode: str) -> list[dict]:
    """All candidate instance entries for one (ref_threshold, tta) combo (no conf filter yet)."""
    items = []
    for item in refined:
        probability = choose_probability(item["prob_base"], item["prob_tta"], ref_threshold, tta_mode)
        x0, y0, x1, y1 = item["bounds"]
        width, height = x1 - x0, y1 - y0
        if width <= 0 or height <= 0:
            continue
        prob_resized = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
        crop_mask = prob_resized >= ref_threshold
        if not crop_mask.any():
            crop_mask = cv2.resize(
                item["seed_crop"].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        entry = entries_from_submask(crop_mask, x0, y0)
        if entry is None:
            continue
        ref_conf = float(prob_resized[crop_mask].mean()) if crop_mask.any() else 0.0
        items.append(
            {
                "entry": entry,
                "det": float(item["score"]),
                "ref_conf": ref_conf,
                "combo": float(item["score"]) * ref_conf,
            }
        )
    return items


def config_entries(
    seed_items: list[dict],
    conf_threshold: float,
    min_area: int,
    morph: str,
    order: str,
    min_area_post: int | None = None,
):
    """Apply conf filter, morphology, score ordering, overlap resolution and min-area."""
    prepared = []
    for item in seed_items:
        if item["det"] < conf_threshold:
            continue
        entry = morph_entry(item["entry"], morph)
        if entry is None:
            continue
        prepared.append(
            {
                "entry": entry,
                "score": item["combo"] if order == "combo" else item["det"],
                "det": item["det"],
                "ref_conf": item["ref_conf"],
            }
        )
    area_limit = min_area if min_area_post is None else min_area_post
    entries = [item["entry"] for item in prepared]
    scores = [item["score"] for item in prepared]
    kept_entries, kept_scores = resolve_mixed(entries, scores, prepared, area_limit)
    return kept_entries, kept_scores


def resolve_mixed(entries: list[dict], scores: list[float], metas: list[dict], minimum_area: int):
    kept_entries: list[dict] = []
    kept_metas: list[dict] = []
    if not entries:
        return [], []
    for index in np.argsort(scores)[::-1]:
        entry = entries[int(index)]
        if entry_area(entry) < minimum_area:
            continue
        updated = entry
        for previous in kept_entries:
            updated = subtract_entry(updated, previous)
            if updated is None:
                break
        if updated is None or entry_area(updated) < minimum_area:
            continue
        kept_entries.append(updated)
        meta = dict(metas[int(index)])
        meta["score"] = float(scores[int(index)])
        kept_metas.append(meta)
    return kept_entries, kept_metas


# ----------------------------------------------------------------------------
# Prediction cache
# ----------------------------------------------------------------------------


def cache_split_name(fold: int, label_source: str) -> str:
    base = f"fold_{fold}" if fold >= 0 else "full"
    return base if label_source == "medoid" else f"{base}_{label_source}"


def get_split_dirs(fold: int, label_source: str) -> tuple[Path, Path]:
    suffix = "" if label_source == "medoid" else f"_{label_source}"
    detector = MODELS / "yolo" / (("full" if fold < 0 else f"fold_{fold}") + suffix) / "best.pt"
    refiner = MODELS / "refiner" / (("full" if fold < 0 else f"fold_{fold}") + suffix) / "best.pt"
    return detector, refiner


def cache_preds(split: str, files: list[Path], detector, refiner, device: str, force: bool = False, limit: int = 0) -> Path:
    directory = PREDS_ROOT / split
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if limit:
        files = files[:limit]
    for number, image_path in enumerate(files, 1):
        output = directory / f"{image_path.stem}.npz"
        if output.exists() and not force:
            continue
        _, refined = predict_image(image_path, detector, refiner, device)
        arrays = {
            "bounds": np.asarray([item["bounds"] for item in refined], dtype=np.int32).reshape(-1, 4),
            "scores": np.asarray([item["score"] for item in refined], dtype=np.float32),
            "prob_base": np.asarray(
                [np.clip(item["prob_base"] * 255.0, 0, 255).astype(np.uint8) for item in refined], dtype=np.uint8
            ).reshape(-1, PROB_CROP, PROB_CROP),
            "prob_tta": np.asarray(
                [np.clip(item["prob_tta"] * 255.0, 0, 255).astype(np.uint8) for item in refined], dtype=np.uint8
            ).reshape(-1, PROB_CROP, PROB_CROP),
            "seed_crop": np.asarray(
                [item["seed_crop"].astype(np.uint8) for item in refined], dtype=np.uint8
            ).reshape(-1, PROB_CROP, PROB_CROP),
        }
        np.savez_compressed(output, **arrays)
        if number % 20 == 0:
            log(f"cache-preds {split}: {number}/{len(files)} images ({time.time()-started:.0f}s)")
    log(f"cache-preds {split}: done in {time.time()-started:.0f}s -> {directory}")
    return directory


def load_cached_preds(directory: Path, stem: str) -> list[dict]:
    path = directory / f"{stem}.npz"
    if not path.exists():
        return []
    payload = np.load(path)
    refined = []
    for index in range(len(payload["scores"])):
        refined.append(
            {
                "bounds": tuple(int(value) for value in payload["bounds"][index]),
                "score": float(payload["scores"][index]),
                "prob_base": payload["prob_base"][index].astype(np.float32) / 255.0,
                "prob_tta": payload["prob_tta"][index].astype(np.float32) / 255.0,
                "seed_crop": payload["seed_crop"][index].astype(bool),
            }
        )
    return refined


# ----------------------------------------------------------------------------
# Data plumbing
# ----------------------------------------------------------------------------


def load_all_indices():
    from train_yolo_seg import choose_medoid_annotators

    coco = load_coco()
    indices = build_indices(coco)
    indices["image_paths"] = {filename: str(TRAIN_IMAGES / filename) for filename in indices["image_ids_by_file"]}
    filenames = sorted(indices["image_ids_by_file"])
    medoid = choose_medoid_annotators(indices)
    folds, fold_source = load_grouped_folds(filenames)
    return indices, medoid, filenames, folds, fold_source


def gt_entry_sets_factory(indices: dict):
    cache: dict[str, list[list[dict]]] = {}

    def get(filename: str) -> list[list[dict]]:
        if filename in cache:
            return cache[filename]
        result = []
        for image_id in indices["image_ids_by_file"][filename]:
            info = indices["images_by_id"][image_id]
            height, width = int(info["height"]), int(info["width"])
            entries = []
            for annotation in indices["annotations_by_image"].get(image_id, []):
                if annotation.get("iscrowd", 0):
                    continue
                entry = entry_from_rle(annotation_to_rle(annotation, height, width))
                if entry is not None:
                    entries.append(entry)
            result.append(entries)
        cache[filename] = result
        return result

    return get


def files_for_fold(fold: int, folds: dict[str, int]) -> list[str]:
    return [filename for filename in sorted(folds) if folds[filename] == fold]


def persist_predictions(output_dir: Path, files: list[str], refined_map: dict, seed_items_map: dict, config: dict, indices: dict):
    """Write pred_rles.csv (for the official metric) + pred_instances.csv (for integration)."""
    pred_rows = []
    instance_rows = []
    for filename in files:
        stem = Path(filename).stem
        entries, metas = config_entries(
            seed_items_map[stem],
            config["conf"],
            config["min_area"],
            config["morph"],
            config["order"],
        )
        for index, (entry, meta) in enumerate(zip(entries, metas), 1):
            full = entry_to_full(entry)
            rle = mask_utils.encode(np.asfortranarray(full.astype(np.uint8)))
            pred_rows.append(
                {"filament_id": f"{stem}_{index}", "segmentation_rle": rle["counts"].decode("ascii")}
            )
            instance_rows.append(
                {
                    "filament_id": f"{stem}_{index}",
                    "image": stem,
                    "rank": index,
                    "det_score": meta["det"],
                    "refiner_conf": meta["ref_conf"],
                    "score": meta["score"],
                    "area": entry_area(entry),
                }
            )
    pred_df = pd.DataFrame(pred_rows, columns=["filament_id", "segmentation_rle"])
    pred_df.to_csv(output_dir / "pred_rles.csv", index=False)
    pd.DataFrame(instance_rows).to_csv(output_dir / "pred_instances.csv", index=False)
    return pred_df


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------


def command_cache_preds(args, indices, folds, device) -> None:
    label_source = args.label_source
    detector_path, refiner_path = get_split_dirs(args.fold, label_source)
    detector = load_detector(Path(args.yolo) if args.yolo else detector_path)
    refiner = load_refiner(Path(args.refiner) if args.refiner else refiner_path, device)
    if args.test:
        files = sorted(TEST_IMAGES.glob("*.jpeg")) + sorted(TEST_IMAGES.glob("*.jpg"))
        unique, seen = [], set()
        for path in files:
            if path.name not in seen:
                seen.add(path.name)
                unique.append(path)
        split = "test_all" if label_source == "all" else "test_medoid"
    else:
        split = cache_split_name(args.fold, label_source)
        unique = [Path(indices["image_paths"][filename]) for filename in files_for_fold(args.fold, folds)]
    cache_preds(split, unique, detector, refiner, device, force=args.force, limit=args.limit)


def command_sweep(args, indices, folds, gt_sets) -> dict:
    label_source = args.label_source
    split = cache_split_name(args.fold, label_source)
    preds_dir = PREDS_ROOT / split
    files = files_for_fold(args.fold, folds)
    if args.limit:
        files = files[: args.limit]
    started = time.time()
    combos = []
    for conf_threshold in CONF_GRID:
        for ref_threshold in REF_GRID:
            for morph in MORPH_GRID:
                for min_area in MIN_AREA_GRID:
                    combos.append(
                        {
                            "conf": conf_threshold,
                            "ref": ref_threshold,
                            "morph": morph,
                            "min_area": min_area,
                            "order": "combo",
                            "tta": "guard",
                        }
                    )
    rows = {index: {"tp": 0, "sum_iou": 0.0, "fp": 0, "fn": 0, "n_pred": 0, "n_gt": 0} for index in range(len(combos))}
    ref_tta_cache: dict[tuple, dict[str, list[dict]]] = {}
    per_image = []
    for number, filename in enumerate(files, 1):
        stem = Path(filename).stem
        refined = load_cached_preds(preds_dir, stem)
        gt_entry_sets = gt_sets(filename)
        n_gt = sum(len(entries) for entries in gt_entry_sets)
        for combo_index, config in enumerate(combos):
            key = (config["ref"], config["tta"])
            if key not in ref_tta_cache:
                ref_tta_cache[key] = {}
            if stem not in ref_tta_cache[key]:
                ref_tta_cache[key][stem] = seed_entries_for(refined, config["ref"], config["tta"])
            entries, metas = config_entries(
                ref_tta_cache[key][stem], config["conf"], config["min_area"], config["morph"], config["order"]
            )
            tp, sum_iou, fp, fn = pq_accumulate(gt_entry_sets, entries)
            row = rows[combo_index]
            row["tp"] += tp
            row["sum_iou"] += sum_iou
            row["fp"] += fp
            row["fn"] += fn
            row["n_pred"] += len(entries)
            row["n_gt"] += n_gt
        per_image.append({"filename": filename, "n_seeds": len(refined), "n_gt": n_gt})
        if number % 20 == 0:
            log(f"sweep {split}: {number}/{len(files)} files ({time.time()-started:.0f}s)")
    table = []
    for index, config in enumerate(combos):
        row = rows[index]
        metrics = finalize_pq(row["tp"], row["sum_iou"], row["fp"], row["fn"])
        table.append({**config, "n_pred": row["n_pred"], "n_gt": row["n_gt"], **metrics})
    table.sort(key=lambda item: -item["pq"])
    output_dir = MODELS / "eval" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(output_dir / "sweep.csv", index=False)
    summary = {
        "split": split,
        "files": len(files),
        "seconds": round(time.time() - started, 1),
        "n_seeds_mean": float(np.mean([item["n_seeds"] for item in per_image])) if per_image else 0.0,
        "n_gt_mean": float(np.mean([item["n_gt"] for item in per_image])) if per_image else 0.0,
        "best": table[0] if table else None,
        "top": table[:20],
    }
    (output_dir / "sweep.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"sweep {split}: best {json.dumps(summary['best'])}")
    return summary


def load_harness_metric():
    try:
        import pq_metric  # local src/pq_metric.py

        return pq_metric
    except Exception:  # noqa: BLE001
        return None


def command_evaluate(args, indices, folds, gt_sets) -> dict:
    label_source = args.label_source
    split = cache_split_name(args.fold, label_source)
    preds_dir = PREDS_ROOT / split
    files = files_for_fold(args.fold, folds)
    if args.limit:
        files = files[: args.limit]
    output_dir = MODELS / "eval" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = {"conf": args.conf, "ref": args.ref, "morph": args.morph, "min_area": args.min_area, "order": args.order, "tta": args.tta}
    seed_items_map = {}
    refined_count = 0
    for number, filename in enumerate(files, 1):
        stem = Path(filename).stem
        refined = load_cached_preds(preds_dir, stem)
        refined_count += len(refined)
        seed_items_map[stem] = seed_entries_for(refined, args.ref, args.tta)
        if number % 20 == 0:
            log(f"evaluate {split}: prepared {number}/{len(files)}")
    pred_df = persist_predictions(output_dir, files, None, seed_items_map, config, indices)

    totals = {"tp": 0, "sum_iou": 0.0, "fp": 0, "fn": 0}
    per_image_rows = []
    for filename in files:
        stem = Path(filename).stem
        gt_entry_sets = gt_sets(filename)
        entries, metas = config_entries(seed_items_map[stem], args.conf, args.min_area, args.morph, args.order)
        tp, sum_iou, fp, fn = pq_accumulate(gt_entry_sets, entries)
        for key, value in (("tp", tp), ("sum_iou", sum_iou), ("fp", fp), ("fn", fn)):
            totals[key] += value
        per_image_rows.append(
            {
                "filename": filename,
                "fold": args.fold,
                "n_gt_annotators": len(gt_entry_sets),
                "n_gt": sum(len(item) for item in gt_entry_sets),
                "n_pred": len(entries),
                **finalize_pq(tp, sum_iou, fp, fn),
            }
        )
    overall = finalize_pq(totals["tp"], totals["sum_iou"], totals["fp"], totals["fn"])
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(output_dir / "per_image.csv", index=False)

    official = None
    metric_module = load_harness_metric()
    if metric_module is not None:
        gt_df = metric_module.gt_df_from_coco(
            TRAIN_IMAGES.parent / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
            stems=[Path(filename).stem for filename in files],
        )
        result = metric_module.evaluate(gt_df, pred_df)
        official = {key: value for key, value in result.items() if key not in ("per_image", "tp_ious")}
        (output_dir / "val_report_official.json").write_text(json.dumps(official, indent=1), encoding="utf-8")
        print(metric_module.format_report(result), flush=True)

    summary = {
        "split": split,
        "fold": args.fold,
        "config": config,
        "files_evaluated": len(per_image_df),
        "seconds": round(time.time() - started, 1),
        **overall,
        "official_pq_metric": official,
        "mean_image_pq": float(per_image_df["pq"].mean()) if len(per_image_df) else 0.0,
        "mean_instances_per_image": float(per_image_df["n_pred"].mean()) if len(per_image_df) else 0.0,
        "mean_gt_instances_per_annotator_image": float(
            per_image_df["n_gt"].sum() / max(per_image_df["n_gt_annotators"].sum(), 1)
        )
        if len(per_image_df)
        else 0.0,
        "cached_seeds_total": refined_count,
    }
    (output_dir / "evaluate.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"EVALUATE {split}: {json.dumps(summary)}")
    return summary


def command_predict_test(args, device) -> Path:
    label_source = args.label_source
    split = "test_all" if label_source == "all" else "test_medoid"
    preds_dir = PREDS_ROOT / split
    test_files = sorted(TEST_IMAGES.glob("*.jpeg")) + sorted(TEST_IMAGES.glob("*.jpg"))
    unique, seen = [], set()
    for path in test_files:
        if path.name not in seen:
            seen.add(path.name)
            unique.append(path)
    if args.limit:
        unique = unique[: args.limit]
    rows = []
    instance_rows = []
    overlap_errors = 0
    started = time.time()
    for number, image_path in enumerate(unique, 1):
        refined = load_cached_preds(preds_dir, image_path.stem)
        seed_items = seed_entries_for(refined, args.ref, args.tta)
        entries, metas = config_entries(seed_items, args.conf, args.min_area, args.morph, args.order)
        occupied = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        for index, (entry, meta) in enumerate(zip(entries, metas), 1):
            full = entry_to_full(entry)
            if np.any(np.logical_and(full, occupied)):
                overlap_errors += 1
            occupied |= full
            rle = mask_utils.encode(np.asfortranarray(full.astype(np.uint8)))
            rows.append({"filament_id": f"{image_path.stem}_{index}", "segmentation_rle": rle["counts"].decode("ascii")})
            instance_rows.append(
                {
                    "filament_id": f"{image_path.stem}_{index}",
                    "image": image_path.stem,
                    "rank": index,
                    "det_score": meta["det"],
                    "refiner_conf": meta["ref_conf"],
                    "score": meta["score"],
                    "area": entry_area(entry),
                }
            )
        if number % 20 == 0:
            log(f"predict-test {number}/{len(unique)} rows={len(rows)} ({time.time()-started:.0f}s)")
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    if submission.empty:
        raise RuntimeError("empty submission")
    if not submission["filament_id"].is_unique:
        raise RuntimeError("duplicate filament_id")
    submission.to_csv(output_path, index=False)
    instance_df = pd.DataFrame(instance_rows)
    instance_df.to_csv(output_path.with_name(output_path.stem + "_instances.csv"), index=False)
    n_images = int(submission["filament_id"].str.rsplit("_", n=1).str[0].nunique())
    summary = {
        "rows": len(submission),
        "images_covered": n_images,
        "config": {"conf": args.conf, "ref": args.ref, "morph": args.morph, "min_area": args.min_area, "order": args.order, "tta": args.tta},
        "mean_instances_per_image": float(instance_df.groupby("image").size().mean()) if len(instance_df) else 0.0,
        "mean_confidence": float(instance_df["score"].mean()) if len(instance_df) else 0.0,
        "overlap_errors": overlap_errors,
        "seconds": round(time.time() - started, 1),
        "path": str(output_path),
    }
    output_path.with_suffix(".meta.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"SUBMISSION: {json.dumps(summary)}")
    return output_path


def command_oof(args, indices, folds, gt_sets) -> dict:
    summaries = []
    for fold in [int(value) for value in args.folds.split(",")]:
        split = cache_split_name(fold, args.label_source)
        if not (PREDS_ROOT / split).exists():
            log(f"oof: missing prediction cache {PREDS_ROOT / split}, skipped")
            continue
        args.fold = fold
        summaries.append(command_evaluate(args, indices, folds, gt_sets))
    if not summaries:
        raise RuntimeError("no fold caches found")
    total = {"tp": 0, "sum_iou": 0.0, "fp": 0, "fn": 0}
    for row in summaries:
        for key in total:
            total[key] += row[key]
    overall = finalize_pq(total["tp"], total["sum_iou"], total["fp"], total["fn"])
    output_dir = MODELS / "eval"
    pd.DataFrame(summaries).to_csv(output_dir / "oof_summary.csv", index=False)
    (output_dir / "oof.json").write_text(json.dumps({"per_fold": summaries, "overall": overall}, indent=1), encoding="utf-8")
    log(f"OOF: {json.dumps(overall)}")
    return overall


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp, need_config: bool = True):
        sp.add_argument("--fold", type=int, default=0)
        sp.add_argument("--label-source", default="all", choices=["all", "medoid"])
        sp.add_argument("--yolo", default=None)
        sp.add_argument("--refiner", default=None)
        sp.add_argument("--device", default="cuda:0")
        sp.add_argument("--limit", type=int, default=0)
        if need_config:
            sp.add_argument("--conf", type=float, default=0.20)
            sp.add_argument("--ref", type=float, default=0.65)
            sp.add_argument("--morph", default="none", choices=["none", "dilate1", "erode1"])
            sp.add_argument("--min-area", type=int, default=50)
            sp.add_argument("--order", default="combo", choices=["combo", "det"])
            sp.add_argument("--tta", default="guard", choices=["none", "guard"])

    cache = sub.add_parser("cache-preds", help="run detector+refiner once and cache all intermediates")
    add_common(cache, need_config=False)
    cache.add_argument("--test", action="store_true")
    cache.add_argument("--force", action="store_true")

    sweep = sub.add_parser("sweep", help="PQ-aware threshold/morphology sweep on cached val predictions")
    add_common(sweep, need_config=False)

    evaluate = sub.add_parser("evaluate", help="evaluate one config (writes pred csv + official PQ report)")
    add_common(evaluate)

    test = sub.add_parser("predict-test", help="write submission csv from cached test predictions")
    add_common(test)
    test.add_argument("--out", default=str(MODELS / "submission_yolo_refiner.csv"))

    oof = sub.add_parser("oof", help="evaluate every available fold prediction cache")
    oof.add_argument("--folds", default="0,1,2,3,4")
    oof.add_argument("--label-source", default="all", choices=["all", "medoid"])
    oof.add_argument("--conf", type=float, default=0.20)
    oof.add_argument("--ref", type=float, default=0.65)
    oof.add_argument("--morph", default="none", choices=["none", "dilate1", "erode1"])
    oof.add_argument("--min-area", type=int, default=50)
    oof.add_argument("--order", default="combo", choices=["combo", "det"])
    oof.add_argument("--tta", default="guard", choices=["none", "guard"])
    oof.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.backends.cudnn.benchmark = True
    device = getattr(args, "device", "cuda:0")

    if args.command in ("cache-preds", "predict-test"):
        holder = read_lock()
        if holder not in (None, "", "modeling"):
            raise RuntimeError(f"GPU_LOCK held by '{holder}'; busy")

    if args.command in ("sweep", "evaluate", "oof"):
        indices, medoid, filenames, folds, fold_source = load_all_indices()
        gt_sets = gt_entry_sets_factory(indices)
        if args.command == "sweep":
            command_sweep(args, indices, folds, gt_sets)
        elif args.command == "evaluate":
            command_evaluate(args, indices, folds, gt_sets)
        else:
            command_oof(args, indices, folds, gt_sets)
        return 0

    if args.command == "predict-test":
        command_predict_test(args, device)
        return 0

    if args.command == "cache-preds":
        indices, medoid, filenames, folds, fold_source = load_all_indices()
        command_cache_preds(args, indices, folds, device)
        return 0

    raise ValueError(args.command)


if __name__ == "__main__":
    sys.exit(main())