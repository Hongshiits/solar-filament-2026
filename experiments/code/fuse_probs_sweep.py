"""Weighted probability-map fusion sweep with official pooled PQ.

Given 1..N probability dumps (uint8 2048x2048, as produced by prob_ensemble /
the dariush/multi-scale/semantic cloud dumps), this script:
  * fuses them with weights x fuse-mode (mean / wmean / max / median),
  * applies the KEEP rule, and scores fold-N validation with the official PQ
    definition (fast bbox-local IoU screening; top-3 re-verified with pq_metric),
  * keeps the best fused probability dump + submission-format CSV,
  * can also fuse test dumps into a submission CSV (--eval-fold -1).

Usage
-----
# evaluate fusion on fold0 val (base dump gets weight 1.0, extras swept)
python src/fuse_probs_sweep.py --prob-npz work/cloud/runs/probs_dump_v1/probs_dariush_fold0.npz     work/cloud/runs/probs_ms_fold0/probs.npz --extra-weights 0.2,0.3,0.5,0.7,1.0 --fuse mean,wmax,max

# build a submission from test dumps
python src/fuse_probs_sweep.py --prob-npz a.npz b.npz --weights 1.0 0.5 --fuse mean --eval-fold -1 \
    --out submissions/fused.csv
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
from eval_semantic import load_gt_entries, pq_accumulate  # noqa: E402

H = W = 2048
DEFAULT_KEEP = dict(E.DEFAULT_KEEP)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def entries_for_prob(prob_u8: np.ndarray, keep: dict):
    comp = E.keep_rule_components(prob_u8, keep)
    if comp is None:
        return []
    lab, cstats, kept = comp
    entries = []
    for cid in kept:
        x = int(cstats[cid, 0])
        y = int(cstats[cid, 1])
        w = int(cstats[cid, 2])
        h = int(cstats[cid, 3])
        sub = lab[y : y + h, x : x + w] == cid
        ys, xs = np.where(sub)
        if len(xs) == 0:
            continue
        bx0, by0 = int(xs.min()), int(ys.min())
        bx1, by1 = int(xs.max()) + 1, int(ys.max()) + 1
        entries.append({"x0": x + bx0, "y0": y + by0, "m": sub[by0:by1, bx0:bx1].astype(np.uint8)})
    return entries


def pred_df_for(entries_by_stem: dict, stems: list[str]) -> pd.DataFrame:
    rows = []
    for stem in stems:
        for index, entry in enumerate(entries_by_stem.get(stem, []), 1):
            full = np.zeros((H, W), dtype=np.uint8)
            full[entry["y0"] : entry["y0"] + entry["m"].shape[0], entry["x0"] : entry["x0"] + entry["m"].shape[1]] = entry["m"]
            counts = mask_utils.encode(np.asfortranarray(full))["counts"]
            rows.append({"filament_id": f"{stem}_{index}", "segmentation_rle": counts.decode("ascii")})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def consensus_gt_df(stems: list[str]) -> pd.DataFrame | None:
    """GT dataframe made of the annotator-majority consensus instances (for PQ')."""
    try:
        import consensus_exp as C  # harness tool
    except Exception:  # noqa: BLE001
        return None
    physical_path = ROOT / "work" / "cache" / "physical.json"
    if not physical_path.exists():
        return None
    physical = json.load(open(physical_path, encoding="utf-8"))["physical"]
    rows = []
    for stem in stems:
        record = physical.get(stem)
        if not record:
            continue
        labels = C.consensus_target_label(record["records"], min_area=20)
        for label_id in np.unique(labels):
            if label_id == 0:
                continue
            inst = labels == label_id
            counts = mask_utils.encode(np.asfortranarray(inst.astype(np.uint8)))["counts"]
            rows.append({"filament_id": f"{stem}_{label_id}", "segmentation_rle": counts.decode("ascii")})
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prob-npz", nargs="+", required=True)
    parser.add_argument("--extra-weights", default="0.2,0.3,0.5,0.7,1.0",
                        help="one weight per extra dump (same for all extras)")
    parser.add_argument("--extra-weight-combos", default=None,
                        help="explicit combos for extra dumps, e.g. '0.1,0.1;0.2,0.1;0.3,0.2' (one per semicolon)")
    parser.add_argument("--fuse", default="mean,wmean,max,median")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--eval-fold", type=int, default=None,
                        help="fold to evaluate (default = --fold); -1 = no eval, just build the submission")
    parser.add_argument("--weights", nargs="*", type=float, default=None, help="explicit weights for the final fusion")
    parser.add_argument("--out", default=None, help="submission csv for --eval-fold -1")
    parser.add_argument("--keep", nargs=4, type=float, default=[DEFAULT_KEEP["t_lo"], DEFAULT_KEEP["t_hi"], DEFAULT_KEEP["min_area"], DEFAULT_KEEP["min_meanp"]])
    parser.add_argument("--out-dir", default=str(ROOT / "work" / "models" / "prob_fusion"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gt-json", default=str(ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    parser.add_argument("--folds-json", default=str(ROOT / "work" / "cache" / "folds.json"))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = dict(t_lo=args.keep[0], t_hi=args.keep[1], min_area=int(args.keep[2]), min_meanp=args.keep[3])
    dumps = [E.load_prob_npz(path) for path in args.prob_npz]
    log(f"loaded {len(dumps)} dumps: {[len(d) for d in dumps]} maps each")

    if args.eval_fold == -1:
        # pure submission build
        weights = args.weights if args.weights else [1.0] * len(dumps)
        fused = E.fuse_probs(dumps, weights=weights, mode=args.fuse.split(",")[0]) if len(dumps) > 1 else dumps[0]
        out = Path(args.out or (ROOT / "submissions" / "fused_probs.csv"))
        out.parent.mkdir(parents=True, exist_ok=True)
        result = E.prob_to_submission(fused, keep=keep, verbose=True)
        if isinstance(result, tuple):
            df = result[0]
        elif isinstance(result, pd.DataFrame):
            df = result
        else:
            df = pd.DataFrame(result, columns=["filament_id", "segmentation_rle"])
        df = df[["filament_id", "segmentation_rle"]]
        df.to_csv(out, index=False)
        log(f"wrote submission {len(df)} rows -> {out}")
        return 0

    folds = json.load(open(args.folds_json, encoding="utf-8"))
    eval_fold = args.fold if args.eval_fold is None else args.eval_fold
    val_stems = sorted(folds["folds"][str(eval_fold)])
    stems = sorted(set(dumps[0].keys()) & set(val_stems))
    if args.limit:
        stems = stems[: args.limit]
    log(f"evaluating fold{eval_fold}: {len(stems)} stems")
    gt_entries = load_gt_entries(Path(args.gt_json), stems)
    log("GT entries built")

    if args.extra_weight_combos:
        extra_combos = [
            [float(value) for value in combo.split(",") if value != ""]
            for combo in args.extra_weight_combos.split(";")
            if combo.strip()
        ]
        for combo in extra_combos:
            if len(combo) != len(dumps) - 1:
                raise ValueError(f"extra weight combo {combo} does not match {len(dumps)-1} extra dumps")
    else:
        extra_combos = None
    weights_grid = [1.0] + [float(v) for v in args.extra_weights.split(",") if v]
    fuse_modes = args.fuse.split(",")
    results = []
    best = None
    stage = time.time()
    for mode in fuse_modes:
        combos = extra_combos if extra_combos else ([[w] * (len(dumps) - 1) for w in weights_grid] if len(dumps) > 1 else [[ ]])
        for combo in combos:
            weights = [1.0] + list(combo)
            fused = E.fuse_probs(dumps, weights=weights, mode=mode) if len(dumps) > 1 else dumps[0]
            entries_by_stem = {}
            totals = {"tp": 0, "sum_iou": 0.0, "fp": 0, "fn": 0, "n_pred": 0}
            for stem in stems:
                entries = entries_for_prob(fused[stem], keep)
                entries_by_stem[stem] = entries
                metrics = pq_accumulate(gt_entries[stem], entries)
                totals["tp"] += metrics["tp"]
                totals["fp"] += metrics["fp"]
                totals["fn"] += metrics["fn"]
                totals["n_pred"] += metrics["n_pred"]
                totals["sum_iou"] += metrics["sq"] * metrics["tp"]
            denominator = totals["tp"] + 0.5 * totals["fp"] + 0.5 * totals["fn"]
            row = {
                "fuse": mode,
                "weights": ",".join(f"{w:g}" for w in weights),
                "pq": totals["sum_iou"] / denominator if denominator else 0.0,
                "sq": totals["sum_iou"] / totals["tp"] if totals["tp"] else 0.0,
                "rq": totals["tp"] / denominator if denominator else 0.0,
                "tp": totals["tp"],
                "fp": totals["fp"],
                "fn": totals["fn"],
                "n_pred": totals["n_pred"],
            }
            results.append(row)
            log(f"{mode:6s} weights={row['weights']}: PQ={row['pq']:.6f} SQ={row['sq']:.6f} RQ={row['rq']:.6f} TP={row['tp']} FP={row['fp']} FN={row['fn']} n_pred={row['n_pred']}")
            candidate = {"row": row, "weights": weights, "mode": mode, "entries": entries_by_stem}
            if best is None or row["pq"] > best["row"]["pq"]:
                best = candidate
    results.sort(key=lambda row: -row["pq"])
    pd.DataFrame(results).to_csv(out_dir / "fusion_sweep.csv", index=False)
    log(f"fusion sweep done in {time.time()-stage:.0f}s; BEST screened: {json.dumps(results[0])}")

    # official verification for top-3 (+ consensus-target PQ' when available)
    gt_df = P.gt_df_from_coco(Path(args.gt_json), stems=stems)
    gt_consensus = consensus_gt_df(stems)
    log(f"consensus GT for PQ': {'yes' if gt_consensus is not None else 'unavailable'}")
    official = []
    best_official = None
    best_pred = None
    for row in results[:3]:
        # rebuild entries for this exact config
        weights = [float(v) for v in row["weights"].split(",")]
        fused = E.fuse_probs(dumps, weights=weights, mode=row["fuse"]) if len(dumps) > 1 else dumps[0]
        entries_by_stem = {stem: entries_for_prob(fused[stem], keep) for stem in stems}
        pred = pred_df_for(entries_by_stem, stems)
        res = P.evaluate(gt_df, pred)
        entry = {"fuse": row["fuse"], "weights": row["weights"], "pq": res["pq"], "sq": res["sq"],
                 "rq": res["rq"], "tp": res["tp"], "fp": res["fp"], "fn": res["fn"], "n_rows": len(pred)}
        if gt_consensus is not None:
            try:
                res_cons = P.evaluate(gt_consensus, pred)
                entry["pq_consensus"] = res_cons["pq"]
                entry["sq_consensus"] = res_cons["sq"]
                entry["rq_consensus"] = res_cons["rq"]
            except Exception as error:  # noqa: BLE001
                log(f"consensus PQ' failed: {error}")
        official.append(entry)
        log(f"OFFICIAL {row['fuse']} weights={row['weights']}: PQ={res['pq']:.6f} SQ={res['sq']:.6f} RQ={res['rq']:.6f} TP={res['tp']} FP={res['fp']} FN={res['fn']}")
        if best_official is None or res["pq"] > best_official["pq"]:
            best_official = official[-1]
            best_pred = pred
            best_weights = weights
            best_mode = row["fuse"]
    if best_pred is not None:
        best_pred.to_csv(out_dir / "best_fused_pred.csv", index=False)
        fused_best = E.fuse_probs(dumps, weights=best_weights, mode=best_mode) if len(dumps) > 1 else dumps[0]
        E.save_prob_npz(fused_best, out_dir / "best_fused_probs.npz")
    payload = {
        "prob_npz": args.prob_npz,
        "eval_fold": eval_fold,
        "keep": keep,
        "dumps": [len(d) for d in dumps],
        "best_screened": results[0],
        "official_verify": official,
        "best_official": best_official,
        "seconds": round(time.time() - stage, 1),
    }
    (out_dir / "fusion_results.json").write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    log(f"BEST official: {json.dumps(best_official)}")
    log(f"results -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
