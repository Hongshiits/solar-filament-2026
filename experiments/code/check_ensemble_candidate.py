"""Validate a notoverkil-ensemble style submission candidate (schema / coverage / overlap / areas).

Checks
------
* required columns exactly filament_id + segmentation_rle, no NaN/duplicates
* every RLE decodes to 2048x2048, area > 0, ids "<stem>_<k>" with a known stem
  (train stems from folds.json or test stems from data/test/test_images)
* per-image: instance count, mask area distribution, pixel coverage, strict
  per-image non-overlap (union area == sum of areas)
* optional comparison with a reference submission (e.g. dariush): shared stems,
  per-image instance counts, area percentiles, matched-mask IoU stats

    python src/check_ensemble_candidate.py \
        --submission C:/Users/ptcu_/filament_assets/downloads/nk_test/submission.csv \
        --reference submissions/dariush_unet_full_v1.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parent.parent
H = W = 2048


def log(message: str) -> None:
    print(message, flush=True)


def load_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    problems = []
    if list(frame.columns) != ["filament_id", "segmentation_rle"]:
        problems.append(f"columns are {list(frame.columns)}, expected ['filament_id', 'segmentation_rle']")
    if frame.isna().any().any():
        problems.append("NaN values present")
    if frame["filament_id"].duplicated().any():
        problems.append(f"duplicate filament_id: {int(frame['filament_id'].duplicated().sum())}")
    if problems:
        raise ValueError("; ".join(problems))
    return frame


def decode_masks(frame: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    per_image: dict[str, list[np.ndarray]] = {}
    for filament_id, rle_text in zip(frame["filament_id"], frame["segmentation_rle"]):
        if "_" not in filament_id:
            raise ValueError(f"bad filament_id {filament_id!r}")
        stem = filament_id.rsplit("_", 1)[0]
        counts = rle_text.encode("ascii") if isinstance(rle_text, str) else rle_text
        mask = mask_utils.decode({"size": [H, W], "counts": counts}).astype(bool)
        if mask.sum() == 0:
            raise ValueError(f"empty mask in {filament_id!r}")
        per_image.setdefault(stem, []).append(mask)
    return per_image


def area_percentiles(areas: list[int]) -> dict:
    if not areas:
        return {}
    arr = np.asarray(areas, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def image_stats(per_image: dict[str, list[np.ndarray]]) -> dict:
    counts = []
    areas = []
    coverage = []
    overlaps = 0
    for stem, masks in per_image.items():
        counts.append(len(masks))
        stack = np.zeros((H, W), dtype=np.uint16)
        for mask in masks:
            stack += mask.astype(np.uint16)
            areas.append(int(mask.sum()))
        overlap_pixels = int(np.max(stack) > 1)
        overlaps += overlap_pixels
        coverage.append(float((stack > 0).mean()))
    return {
        "n_images": len(per_image),
        "n_instances": int(sum(counts)),
        "instances_per_image": {
            "min": int(np.min(counts)),
            "median": float(np.median(counts)),
            "max": int(np.max(counts)),
            "mean": float(np.mean(counts)),
        },
        "mask_area": area_percentiles(areas),
        "coverage_fraction": {
            "min": float(np.min(coverage)),
            "median": float(np.median(coverage)),
            "max": float(np.max(coverage)),
        },
        "overlap_pixels_total": overlaps,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--folds-json", default=str(ROOT / "work" / "cache" / "folds.json"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    frame = load_submission(Path(args.submission))
    per_image = decode_masks(frame)
    folds = json.load(open(args.folds_json, encoding="utf-8"))
    train_stems = set(folds["folds"]["0"]) | set(folds["folds"]["1"]) | set(folds["folds"]["2"]) | set(folds["folds"]["3"]) | set(folds["folds"]["4"])
    test_dir = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
    test_stems = {path.stem for path in test_dir.glob("*.jpeg")} if test_dir.exists() else set()
    unknown = [stem for stem in per_image if stem not in train_stems and stem not in test_stems]
    report = {
        "submission": str(args.submission),
        "rows": int(len(frame)),
        "images": len(per_image),
        "kind": "test" if not unknown and set(per_image) <= test_stems else ("fold" if set(per_image) <= train_stems else "mixed"),
        "unknown_stems": unknown[:10],
        "stats": image_stats(per_image),
    }
    log(json.dumps(report, indent=1))
    if args.reference:
        reference_frame = load_submission(Path(args.reference))
        reference = decode_masks(reference_frame)
        shared = sorted(set(per_image) & set(reference))
        matched_iou = []
        added = 0
        missing = 0
        for stem in shared:
            masks_a = per_image[stem]
            masks_b = reference[stem]
            rles_a = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in masks_a]
            rles_b = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in masks_b]
            iou = mask_utils.iou(rles_a, rles_b, [0] * len(rles_b))
            matched_iou.append(float(iou.max()) if iou.size else 0.0)
            added += len(masks_a)
            missing += len(masks_b)
        report["reference"] = {
            "path": str(args.reference),
            "rows": int(len(reference_frame)),
            "images": len(reference),
            "shared_images": len(shared),
            "candidate_instances_in_shared": int(added),
            "reference_instances_in_shared": int(missing),
            "per_image_best_iou_mean": float(np.mean(matched_iou)) if matched_iou else 0.0,
        }
        log(json.dumps(report["reference"], indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
