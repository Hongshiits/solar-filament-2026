"""Evaluate semantic U-Net fold probability dumps (uint8 prob maps) with official PQ.

Used for the cloud-trained consensus semantic U-Net (src/train_semantic.py):
  * baseline = the KEEP rule used by train_semantic (t_lo 0.35 / t_hi 0.99 / min_area 200 / min_meanp 0.85)
  * dense KEEP-rule sweep (t_lo / t_hi / min_area / min_meanp) scored with the official
    pooled PQ definition (fast bbox-local IoU; top configs re-verified with src/pq_metric.py)
  * exports the best fold's predictions as a submission-format CSV for further
    ensembling / refiner experiments.

Usage:
    python src/eval_semantic.py --probs work/cloud/runs/<run>/sem_fold0/val_probs_best.npz
    python src/eval_semantic.py --probs a.npz --val-report work/.../val_report.json --out-dir work/models/sem_fold0_eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import prob_ensemble as E  # noqa: E402
import pq_metric as P  # noqa: E402

H = W = 2048
T_LO_GRID = [0.30, 0.35, 0.40]
T_HI_GRID = [0.95, 0.99]
MIN_AREA_GRID = [100, 200, 400]
MIN_MEANP_GRID = [0.80, 0.85, 0.90]
BASELINE_KEEP = dict(t_lo=0.35, t_hi=0.99, min_area=200, min_meanp=0.85)
AREA_FLOOR = 20


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_gt_entries(json_path: Path, stems: list[str]) -> dict[str, list[list[dict]]]:
    """stem -> [annotator record -> list of bbox-local GT entries]."""
    import json as _json

    coco = _json.load(open(json_path, encoding="utf-8"))
    images = {str(image["id"]): image for image in coco["images"]}
    ann_by_image: dict[str, list[dict]] = {}
    for annotation in coco["annotations"]:
        ann_by_image.setdefault(str(annotation["image_id"]), []).append(annotation)
    image_ids_by_stem: dict[str, list[str]] = {}
    for image_id, image in images.items():
        stem = image_id.split("-", maxsplit=1)[-1]
        image_ids_by_stem.setdefault(stem, []).append(image_id)
    out = {}
    for stem in stems:
        record_sets = []
        for image_id in sorted(image_ids_by_stem.get(stem, [])):
            entries = []
            for annotation in ann_by_image.get(image_id, []):
                if annotation.get("iscrowd", 0):
                    continue
                segmentation = annotation.get("segmentation")
                rles = mask_utils.frPyObjects(segmentation, H, W) if isinstance(segmentation, list) else segmentation
                rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
                mask = mask_utils.decode(rle).astype(bool)
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                x0, y0 = int(xs.min()), int(ys.min())
                entries.append(
                    {"x0": x0, "y0": y0, "m": mask[y0 : int(ys.max()) + 1, x0 : int(xs.max()) + 1].astype(np.uint8)}
                )
            record_sets.append(entries)
        out[stem] = record_sets
    return out


def entries_iou(a: dict, b: dict) -> float:
    x0 = max(a["x0"], b["x0"])
    y0 = max(a["y0"], b["y0"])
    x1 = min(a["x0"] + a["m"].shape[1], b["x0"] + b["m"].shape[1])
    y1 = min(a["y0"] + a["m"].shape[0], b["y0"] + b["m"].shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    va = a["m"][y0 - a["y0"] : y1 - a["y0"], x0 - a["x0"] : x1 - a["x0"]]
    vb = b["m"][y0 - b["y0"] : y1 - b["y0"], x0 - b["x0"] : x1 - b["x0"]]
    inter = int(np.logical_and(va, vb).sum())
    if inter == 0:
        return 0.0
    union = int(a["m"].sum()) + int(b["m"].sum()) - inter
    return inter / union if union > 0 else 0.0


def pq_accumulate(gt_sets, pred_entries) -> dict:
    tp = 0
    sum_iou = 0.0
    fp = 0
    fn = 0
    n_pred = len(pred_entries)
    for gt_entries in gt_sets:
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
        for gi, gt_entry in enumerate(gt_entries):
            for pi, pred_entry in enumerate(pred_entries):
                iou = entries_iou(gt_entry, pred_entry)
                if iou > 0.5:
                    tp += 1
                    sum_iou += iou
                    matched_gt[gi] = True
                    matched_pred[pi] = True
        fp += matched_pred.count(False)
        fn += matched_gt.count(False)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": sum_iou / denominator if denominator else 0.0,
        "sq": sum_iou / tp if tp else 0.0,
        "rq": tp / denominator if denominator else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_pred": n_pred,
    }


def components_for(prob: np.ndarray, t_lo: float):
    out = E.components_with_stats(prob, t_lo, area_floor=AREA_FLOOR, connectivity=8)
    if out is None:
        return None
    lab, cstats, ids, areas, maxp, meanp = out
    return {"lab": lab, "cstats": cstats, "ids": ids, "areas": areas, "maxp": maxp, "meanp": meanp}


def entries_for_config(components_map: dict, keep: dict, stems: list[str]):
    """Build bbox-local entry lists per image for one KEEP configuration."""
    thr_hi = E._thr_u8(keep["t_hi"])
    thr_mean = float(keep["min_meanp"]) * 255.0
    per_image = {}
    for stem in stems:
        comp = components_map.get(stem)
        if comp is None:
            per_image[stem] = []
            continue
        ok = (comp["maxp"] > thr_hi) & (comp["areas"] >= int(keep["min_area"])) & (comp["meanp"] >= thr_mean)
        kept_ids = comp["ids"][ok]
        entries = []
        lab = comp["lab"]
        cstats = comp["cstats"]
        for cid in kept_ids:
            x = int(cstats[cid, 0])
            y = int(cstats[cid, 1])
            w = int(cstats[cid, 2])
            h = int(cstats[cid, 3])
            sub = (lab[y : y + h, x : x + w] == cid)
            ys, xs = np.where(sub)
            if len(xs) == 0:
                continue
            bx0, by0 = int(xs.min()), int(ys.min())
            bx1, by1 = int(xs.max()) + 1, int(ys.max()) + 1
            entries.append({"x0": x + bx0, "y0": y + by0, "m": sub[by0:by1, bx0:bx1].astype(np.uint8)})
        per_image[stem] = entries
    return per_image


def pred_df_for(per_image: dict, stems: list[str]) -> pd.DataFrame:
    rows = []
    for stem in stems:
        for index, entry in enumerate(per_image[stem], 1):
            full = np.zeros((H, W), dtype=np.uint8)
            full[entry["y0"] : entry["y0"] + entry["m"].shape[0], entry["x0"] : entry["x0"] + entry["m"].shape[1]] = entry["m"]
            counts = mask_utils.encode(np.asfortranarray(full))["counts"]
            rows.append({"filament_id": f"{stem}_{index}", "segmentation_rle": counts.decode("ascii")})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probs", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gt-json", default=str(ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    parser.add_argument("--folds-json", default=str(ROOT / "work" / "cache" / "folds.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "work" / "models" / "sem_fold0_eval"))
    parser.add_argument("--val-report", default=None, help="optional train_semantic val_report.json for cross-check")
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    folds = json.load(open(args.folds_json, encoding="utf-8"))
    val_stems = sorted(folds["folds"][str(args.fold)])
    prob = E.load_prob_npz(args.probs)
    stems = sorted(set(prob.keys()) & set(val_stems))
    if args.limit:
        stems = stems[: args.limit]
    log(f"probs: {len(prob)} maps, using fold{args.fold} val stems: {len(stems)}")
    gt_entries = load_gt_entries(Path(args.gt_json), stems)
    log(f"GT entries built ({time.time()-started:.0f}s)")

    results = []
    # baseline (train_semantic KEEP rule)
    t_lo_values = [BASELINE_KEEP["t_lo"]] + [t for t in T_LO_GRID if t != BASELINE_KEEP["t_lo"]]
    components_map: dict[float, dict] = {}
    stage = time.time()
    for t_lo in t_lo_values:
        components_map[t_lo] = {stem: components_for(prob[stem], t_lo) for stem in stems}
        log(f"components t_lo={t_lo:.2f} done ({time.time()-stage:.0f}s)")

    def score(keep: dict) -> dict:
        per_image = {}
        for stem in stems:
            comp = components_map[keep["t_lo"]].get(stem)
            if comp is None:
                per_image[stem] = []
                continue
            thr_hi = E._thr_u8(keep["t_hi"])
            thr_mean = float(keep["min_meanp"]) * 255.0
            ok = (comp["maxp"] > thr_hi) & (comp["areas"] >= int(keep["min_area"])) & (comp["meanp"] >= thr_mean)
            entries = []
            lab = comp["lab"]
            cstats = comp["cstats"]
            for cid in comp["ids"][ok]:
                x = int(cstats[cid, 0])
                y = int(cstats[cid, 1])
                w = int(cstats[cid, 2])
                h = int(cstats[cid, 3])
                sub = (lab[y : y + h, x : x + w] == cid)
                ys, xs = np.where(sub)
                if len(xs) == 0:
                    continue
                bx0, by0 = int(xs.min()), int(ys.min())
                bx1, by1 = int(xs.max()) + 1, int(ys.max()) + 1
                entries.append({"x0": x + bx0, "y0": y + by0, "m": sub[by0:by1, bx0:bx1].astype(np.uint8)})
            per_image[stem] = entries
        totals = {"tp": 0, "sum_iou": 0.0, "fp": 0, "fn": 0, "n_pred": 0}
        for stem in stems:
            metrics = pq_accumulate(gt_entries[stem], per_image[stem])
            totals["tp"] += metrics["tp"]
            totals["fp"] += metrics["fp"]
            totals["fn"] += metrics["fn"]
            totals["n_pred"] += metrics["n_pred"]
            totals["sum_iou"] += metrics["sq"] * metrics["tp"]
        denominator = totals["tp"] + 0.5 * totals["fp"] + 0.5 * totals["fn"]
        row = {
            **keep,
            "pq": totals["sum_iou"] / denominator if denominator else 0.0,
            "sq": totals["sum_iou"] / totals["tp"] if totals["tp"] else 0.0,
            "rq": totals["tp"] / denominator if denominator else 0.0,
            "tp": totals["tp"],
            "fp": totals["fp"],
            "fn": totals["fn"],
            "n_pred": totals["n_pred"],
        }
        return row, per_image

    baseline_row, baseline_entries = score(dict(BASELINE_KEEP))
    results.append(baseline_row)
    log(f"BASELINE KEEP={BASELINE_KEEP}: PQ={baseline_row['pq']:.6f} SQ={baseline_row['sq']:.6f} RQ={baseline_row['rq']:.6f} TP={baseline_row['tp']} FP={baseline_row['fp']} FN={baseline_row['fn']} n_pred={baseline_row['n_pred']}")

    cache: dict[tuple, dict] = {}
    cache[tuple(sorted(BASELINE_KEEP.items()))] = baseline_entries
    if not args.no_sweep:
        for t_lo in T_LO_GRID:
            for t_hi in T_HI_GRID:
                for min_area in MIN_AREA_GRID:
                    for min_meanp in MIN_MEANP_GRID:
                        keep = dict(t_lo=t_lo, t_hi=t_hi, min_area=min_area, min_meanp=min_meanp)
                        key = tuple(sorted(keep.items()))
                        if key in cache:
                            continue
                        row, per_image = score(keep)
                        cache[key] = per_image
                        results.append(row)
        results.sort(key=lambda row: -row["pq"])
        pd.DataFrame(results).to_csv(out_dir / "keep_sweep.csv", index=False)
        log(f"keep sweep: {len(results)} configs ({time.time()-stage:.0f}s)")

    # official verification (pq_metric) for top-3
    gt_df = P.gt_df_from_coco(Path(args.gt_json), stems=stems)
    official = []
    best_official = None
    best_pred = None
    for row in results[:3]:
        keep = dict(t_lo=row["t_lo"], t_hi=row["t_hi"], min_area=int(row["min_area"]), min_meanp=row["min_meanp"])
        per_image = cache[tuple(sorted(keep.items()))]
        pred = pred_df_for(per_image, stems)
        res = P.evaluate(gt_df, pred)
        official.append(
            {
                "keep": keep,
                "pq": res["pq"],
                "sq": res["sq"],
                "rq": res["rq"],
                "tp": res["tp"],
                "fp": res["fp"],
                "fn": res["fn"],
                "n_rows": len(pred),
            }
        )
        log(f"OFFICIAL {keep}: PQ={res['pq']:.6f} SQ={res['sq']:.6f} RQ={res['rq']:.6f} TP={res['tp']} FP={res['fp']} FN={res['fn']}")
        if best_official is None or res["pq"] > best_official["pq"]:
            best_official = official[-1]
            best_pred = pred
    if best_pred is not None:
        best_pred.to_csv(out_dir / "best_pred.csv", index=False)
    payload = {
        "probs": str(args.probs),
        "fold": args.fold,
        "stems": len(stems),
        "baseline_keep": BASELINE_KEEP,
        "baseline": baseline_row,
        "best_screened": results[0],
        "official_verify": official,
        "val_report_cross_check": json.load(open(args.val_report, encoding="utf-8")) if args.val_report and Path(args.val_report).exists() else None,
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    log(f"BEST official: {json.dumps(best_official)}")
    log(f"results -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
