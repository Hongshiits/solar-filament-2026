"""Generate sem x dariush fusion candidate submissions (+ sem-only), validate and summarise.

Given the dariush test probability dump and the multi-fold sem test probability
average (layout a npz), produce:
  * submissions/candidates/fused_w010.csv ... fused_w050.csv  (wmean fusion, weights [1, w])
  * submissions/candidates/sem_only.csv                       (sem probabilities alone)
and for every file: rows / images / vs-dariush instance deltas / matched-mask IoU
stats / area percentiles, written to submissions/candidates/summary.json.

    python src/make_sem_fusion_candidates.py \
        --dariush-probs work/cloud/runs/probs_dump_v1/probs_dariush_test.npz \
        --sem-probs C:/Users/ptcu_/filament_assets/probs/sem_test_probs_5fold.npz \
        --weights 0.10 0.15 0.20 0.25 0.30 0.40 0.50 --keep 0.35 0.99 200 0.85
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import prob_ensemble as E  # noqa: E402

H = W = 2048


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def decode_csv(path: Path) -> dict[str, list[np.ndarray]]:
    frame = pd.read_csv(path, dtype=str)
    out: dict[str, list[np.ndarray]] = {}
    for filament_id, rle_text in zip(frame["filament_id"], frame["segmentation_rle"]):
        stem = filament_id.rsplit("_", 1)[0]
        counts = rle_text.encode("ascii") if isinstance(rle_text, str) else rle_text
        out.setdefault(stem, []).append(mask_utils.decode({"size": [H, W], "counts": counts}).astype(bool))
    return out


def compare(element: dict[str, list[np.ndarray]], reference: dict[str, list[np.ndarray]]) -> dict:
    shared = sorted(set(element) & set(reference))
    matched_iou = []
    kept = added = dropped = 0
    for stem in shared:
        masks_a = element[stem]
        masks_b = reference[stem]
        if not masks_a or not masks_b:
            continue
        rles_a = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in masks_a]
        rles_b = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in masks_b]
        iou = mask_utils.iou(rles_a, rles_b, [0] * len(rles_b))
        best = iou.max(axis=1) if iou.size else np.zeros(len(masks_a))
        matched_iou.extend(best.tolist())
        kept += int((best > 0.9).sum())
        added += int((best < 0.5).sum())
    return {
        "shared_images": len(shared),
        "instances": int(sum(len(v) for v in element.values())),
        "kept_iou_gt_0.9": kept,
        "new_iou_lt_0.5": added,
        "per_instance_best_iou_mean": float(np.mean(matched_iou)) if matched_iou else 0.0,
    }


def area_stats(element: dict[str, list[np.ndarray]]) -> dict:
    areas = [int(mask.sum()) for masks in element.values() for mask in masks]
    if not areas:
        return {}
    arr = np.asarray(areas, dtype=np.float64)
    return {"min": float(arr.min()), "p25": float(np.percentile(arr, 25)), "median": float(np.median(arr)), "p75": float(np.percentile(arr, 75)), "max": float(arr.max())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dariush-probs", default=str(ROOT / "work" / "cloud" / "runs" / "probs_dump_v1" / "probs_dariush_test.npz"))
    parser.add_argument("--sem-probs", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    parser.add_argument("--keep", nargs=4, type=float, default=[0.35, 0.99, 200, 0.85])
    parser.add_argument("--out-dir", default=str(ROOT / "submissions" / "candidates"))
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--dariush-reference", default=str(ROOT / "submissions" / "dariush_unet_full_v1.csv"))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = dict(t_lo=args.keep[0], t_hi=args.keep[1], min_area=int(args.keep[2]), min_meanp=args.keep[3])
    started = time.time()

    dariush = E.load_prob_npz(args.dariush_probs)
    sem = E.load_prob_npz(args.sem_probs)
    stems = sorted(set(dariush) & set(sem))
    log(f"dariush {len(dariush)} maps, sem {len(sem)} maps, common {len(stems)}")
    if not stems:
        raise ValueError("dariush and sem probs have no common stems")
    dariush_sub = {stem: dariush[stem] for stem in stems}
    sem_sub = {stem: sem[stem] for stem in stems}
    reference = decode_csv(Path(args.dariush_reference))
    summary = {"keep": keep, "stems": len(stems), "fused": [], "sem_only": None}
    result = E.prob_to_submission(dariush_sub, keep=keep, verbose=False)
    baseline = result[0] if isinstance(result, tuple) else result
    log(f"dariush baseline: {len(baseline)} rows / {baseline['filament_id'].str.rsplit(chr(95), n=1).str[0].nunique()} images")

    for weight in args.weights:
        tag = f"w{int(round(weight * 100)):03d}"
        fused = E.fuse_probs([dariush_sub, sem_sub], weights=[1.0, weight], mode="wmean")
        result = E.prob_to_submission(fused, keep=keep, verbose=False)
        frame = result[0] if isinstance(result, tuple) else result
        frame = frame[["filament_id", "segmentation_rle"]]
        out = out_dir / f"fused_{tag}.csv"
        frame.to_csv(out, index=False)
        decoded = decode_csv(out)
        stats = compare(decoded, reference)
        stats.update({"tag": tag, "sem_weight": weight, "rows": len(frame), "images": int(frame["filament_id"].str.rsplit("_", n=1).str[0].nunique()), "area": area_stats(decoded), "path": str(out)})
        code = 0 if args.no_validate else subprocess.call([sys.executable, str(ROOT / "src" / "validate_submission.py"), str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stats["validate_exit"] = int(code)
        summary["fused"].append(stats)
        log(f"{tag}: rows={stats['rows']} images={stats['images']} kept={stats['kept_iou_gt_0.9']} new={stats['new_iou_lt_0.5']} area_median={stats['area'].get('median')} validate={code}")

    result = E.prob_to_submission(sem_sub, keep=keep, verbose=False)
    frame = result[0] if isinstance(result, tuple) else result
    frame = frame[["filament_id", "segmentation_rle"]]
    out = out_dir / "sem_only.csv"
    frame.to_csv(out, index=False)
    decoded = decode_csv(out)
    stats = compare(decoded, reference)
    stats.update({"tag": "sem_only", "sem_weight": 1.0, "rows": len(frame), "images": int(frame["filament_id"].str.rsplit("_", n=1).str[0].nunique()), "area": area_stats(decoded), "path": str(out)})
    code = 0 if args.no_validate else subprocess.call([sys.executable, str(ROOT / "src" / "validate_submission.py"), str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stats["validate_exit"] = int(code)
    summary["sem_only"] = stats
    log(f"sem_only: rows={stats['rows']} images={stats['images']} kept={stats['kept_iou_gt_0.9']} new={stats['new_iou_lt_0.5']} validate={code}")
    summary["seconds"] = round(time.time() - started, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"DONE -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
