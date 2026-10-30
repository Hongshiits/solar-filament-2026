"""Reproduce final submission #2: vote-filtered 15-model YOLO consensus (public LB 0.39).

Inputs
  --unfiltered-csv : consensus output of scripts/../experiments/code/cloud_notoverkil_ensemble.py
                     (assemble mode; 2328 rows over 180 images; one row per consensus instance)
  --support-npz    : assets/nk_v3_consensus_support.npz with arrays 'support' (int) and 'best_ious'
                     (float), aligned row-by-row with the unfiltered CSV (one entry per instance:
                     how many of the 15 models back that instance with IoU > 0.5)
  --threshold      : keep instances supported by at least this many models (15 = unanimous)

Output: filtered submission CSV (filament_id, segmentation_rle), re-indexed per image.

Expected output at --threshold 15: 1259 rows over 177 images, public LB 0.39 (Kaggle ref 57782016).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unfiltered-csv", required=True)
    ap.add_argument("--support-npz", required=True)
    ap.add_argument("--threshold", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.unfiltered_csv, dtype=str)
    z = np.load(args.support_npz)
    support = z["support"]
    if len(df) != len(support):
        raise SystemExit(f"row mismatch: {len(df)} rows vs {len(support)} support entries")
    keep = df[support >= args.threshold].copy()
    # re-index filament ids per image: <stem>_<k>
    out_rows = []
    for stem, grp in keep.groupby(keep["filament_id"].str.rsplit("_", n=1).str[0], sort=True):
        for k, (_, row) in enumerate(grp.iterrows(), 1):
            out_rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": row["segmentation_rle"]})
    out = pd.DataFrame(out_rows, columns=["filament_id", "segmentation_rle"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"kept {len(out)} instances over {out['filament_id'].str.rsplit('_', n=1).str[0].nunique()} images -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
