"""Merge two (or more) submission CSVs into one non-overlapping submission.

Official constraint: within one image, predicted masks must be **pixel-wise
non-overlapping**, otherwise the submission is rejected (SubmissionStatus.ERROR).
Two ensemble members may both predict the same filament, so the merge resolves
overlaps by *score-ordered panoptic painting*:

  1. decode every instance mask of one physical image (all input files);
  2. sort instances by score (descending), ties by input order and row order;
  3. paint them onto an empty canvas; an instance keeps only the pixels that are
     still free -> residual mask;
  4. drop the instance if residual area is 0 or < min_area (default 20 px);
  5. renumber surviving instances per image as "{stem}_{k}".

Every instance may carry an optional confidence column (names configured with
--score-cols, e.g. confidence/score/prob); when absent, input order and
--weights decide the priority.  With --weights 1.0,0.6 the first file wins ties.

Usage
-----
    python src/ensemble.py --inputs work/public070.csv work/models/oof.csv \
        --weights 1.0 0.6 --out submissions/ensemble.csv
    python src/ensemble.py --inputs a.csv b.csv --out merged.csv --min-area 20
    python src/ensemble.py --selftest

Output CSV has exactly the two official columns: filament_id, segmentation_rle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
import pq_metric as P  # noqa: E402

DEFAULT_SCORE_COLS = ("confidence", "score", "prob", "probability", "conf")


def _stem_of(filament_id: str) -> str:
    s = str(filament_id)
    if "_" not in s:
        raise ValueError(f"filament_id without suffix: {s!r}")
    return s.rsplit("_", 1)[0]


def load_submission(path: str | Path, score_cols=DEFAULT_SCORE_COLS):
    """Read a submission CSV -> (DataFrame[filament_id, segmentation_rle], score column name or None)."""
    p = Path(path)
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    missing = {"filament_id", "segmentation_rle"} - set(df.columns)
    if missing:
        raise ValueError(f"{p}: missing columns {sorted(missing)} (got {list(df.columns)[:6]})")
    if len(df) == 0:
        raise ValueError(f"{p}: empty submission")
    score_col = next((c for c in score_cols if c in df.columns), None)
    return df, score_col


def merge_frames(frames, h: int = P.H, w: int = P.W, min_area: int = 20,
                 weights=None, score_cols=DEFAULT_SCORE_COLS,
                 max_residual_frac: float = 0.0, verbose: bool = True):
    """Merge decoded submissions with score-ordered non-overlap resolution.

    Args:
        frames: list of (DataFrame, score_column_name_or_None, weight).
        max_residual_frac: 0.0 (default) keeps every non-empty residual (plain
            panoptic painting).  Set e.g. 0.5 to drop near-duplicates: an instance is
            dropped when its still-free fraction (residual/original) is below this
            value, i.e. when a higher-score instance already covers most of it.
    Returns:
        (out_df, stats dict)
    """
    t0 = time.time()
    if weights is None:
        weights = [1.0] * len(frames)
    if len(weights) != len(frames):
        raise ValueError("weights length != number of inputs")
    per_image: dict[str, list] = defaultdict(list)
    n_in = 0
    for src, ((df, score_col), weight) in enumerate(zip(frames, weights)):
        weight = float(weight)
        rows = []
        for order, row in enumerate(df.itertuples(index=False)):
            fid = str(getattr(row, "filament_id"))
            rle = str(getattr(row, "segmentation_rle"))
            if score_col is not None:
                try:
                    base = float(getattr(row, score_col))
                except Exception:
                    base = 0.0
            else:
                base = 1.0
            rows.append((_stem_of(fid), base * weight, src, order, rle))
        for stem, score, s, o, rle in rows:
            per_image[stem].append((score, s, o, rle))
        n_in += len(df)

    stats = {
        "inputs": [{"rows": int(len(df)), "weight": float(wt),
                    "score_col": sc} for (df, sc), wt in zip(frames, weights)],
        "input_rows": int(n_in),
        "height": int(h), "width": int(w), "min_area": int(min_area),
        "images": 0, "output_rows": 0,
        "dropped_full_overlap": 0, "dropped_small_residual": 0, "dropped_duplicate": 0,
        "max_residual_frac": float(max_residual_frac),
        "pixels_in": 0, "pixels_kept": 0,
        "per_image_dropped": {},
    }
    out_rows = []
    for stem in sorted(per_image):
        items = per_image[stem]
        items.sort(key=lambda t: (-t[0], t[1], t[2]))
        canvas = np.zeros(int(h) * int(w), dtype=bool)
        kept = []
        dropped_full = dropped_small = dropped_dup = 0
        for _score, _src, _order, rle in items:
            try:
                mask = (P.decode_rle(rle, h, w) != 0).reshape(-1)
            except Exception:
                continue
            total = int(np.count_nonzero(mask))
            residual = mask & ~canvas
            area = int(np.count_nonzero(residual))
            if area == 0:
                dropped_full += 1
                continue
            if (float(max_residual_frac) > 0.0 and total > 0
                    and (area / total) < float(max_residual_frac)):
                dropped_dup += 1
                continue
            if area < int(min_area):
                dropped_small += 1
                continue
            canvas |= residual
            kept.append(residual)
            stats["pixels_in"] += total
            stats["pixels_kept"] += area
        stats["images"] += 1
        stats["dropped_full_overlap"] += dropped_full
        stats["dropped_small_residual"] += dropped_small
        stats["dropped_duplicate"] += dropped_dup
        if dropped_full or dropped_small or dropped_dup:
            stats["per_image_dropped"][stem] = {"full": dropped_full, "small": dropped_small,
                                                "duplicate": dropped_dup}
        for k, residual in enumerate(kept, start=1):
            out_rows.append({
                "filament_id": f"{stem}_{k}",
                "segmentation_rle": P.encode_mask(residual.reshape(int(h), int(w))),
            })
    out_df = pd.DataFrame(out_rows, columns=["filament_id", "segmentation_rle"])
    stats["output_rows"] = int(len(out_df))
    stats["kept_ratio"] = round(len(out_df) / n_in, 4) if n_in else 0.0
    stats["pixel_keep_ratio"] = round(stats["pixels_kept"] / stats["pixels_in"], 4) if stats["pixels_in"] else 0.0
    stats["runtime_seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(f"[ensemble] inputs: " + ", ".join(
            f"{s['rows']} rows (w={s['weight']}, score={s['score_col']})" for s in stats["inputs"]))
        print(f"[ensemble] {n_in} -> {len(out_df)} instances ({stats['kept_ratio'] * 100:.1f}%) over "
              f"{stats['images']} images; dropped full-overlap={stats['dropped_full_overlap']}, "
              f"small-residual={stats['dropped_small_residual']}, "
              f"near-duplicate={stats['dropped_duplicate']}; "
              f"pixels kept {stats['pixel_keep_ratio'] * 100:.2f}%")
        print(f"[ensemble] {stats['runtime_seconds']}s")
    return out_df, stats


def merge_files(input_paths, out_path, weights=None, min_area: int = 20,
                h: int = P.H, w: int = P.W, score_cols=DEFAULT_SCORE_COLS,
                max_residual_frac: float = 0.0,
                stats_json: str | None = None, verbose: bool = True) -> dict:
    frames = []
    for p in input_paths:
        df, sc = load_submission(p, score_cols)
        frames.append((df, sc, p))
    merged, stats = merge_frames([(df, sc) for df, sc, _ in frames],
                                 h=h, w=w, min_area=min_area, weights=weights,
                                 score_cols=score_cols, max_residual_frac=max_residual_frac,
                                 verbose=verbose)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    stats["out_path"] = str(out)
    stats["out_bytes"] = int(out.stat().st_size)
    if verbose:
        print(f"[ensemble] wrote {out} ({stats['out_bytes'] / 1e6:.2f} MB)")
    if stats_json:
        Path(stats_json).parent.mkdir(parents=True, exist_ok=True)
        Path(stats_json).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


# --------------------------------------------------------------------------------------
# self-test on synthetic masks
# --------------------------------------------------------------------------------------
def selftest() -> int:
    # NOTE: masks are encoded at the official 2048x2048 size (submission contract)
    hh = ww = P.H
    A = np.zeros((hh, ww), np.uint8); A[0:10, 0:10] = 1
    B = np.zeros((hh, ww), np.uint8); B[0:10, 20:30] = 1
    C = np.zeros((hh, ww), np.uint8); C[20:30, 0:10] = 1
    A2 = np.zeros((hh, ww), np.uint8); A2[0:10, 2:12] = 1
    tiny = np.zeros((hh, ww), np.uint8); tiny[40:41, 40:43] = 1
    df1 = pd.DataFrame({"filament_id": ["imgX_1", "imgX_2"],
                        "segmentation_rle": [P.encode_mask(A), P.encode_mask(B)]})
    df2 = pd.DataFrame({"filament_id": ["imgX_1", "imgX_2", "imgX_3"],
                        "confidence": ["0.9", "0.8", "0.1"],
                        "segmentation_rle": [P.encode_mask(A2), P.encode_mask(C),
                                             P.encode_mask(tiny)]})
    out, stats = merge_frames([(df1, None), (df2, "confidence")],
                              h=hh, w=ww, min_area=20, weights=[1.2, 1.0], verbose=False)
    ms = [P.decode_rle(r) for r in out["segmentation_rle"]]
    canvas = np.zeros((hh, ww), np.uint16)
    for k, m in enumerate(ms, start=1):
        canvas += (m > 0).astype(np.uint16)
    checks = {
        "output rows == 4 (A, B, C, A2-residual; tiny dropped)": len(out) == 4,
        "no pixel overlap": int(canvas.max()) <= 1,
        "ids renumbered per image": list(out["filament_id"]) == [f"imgX_{i}" for i in range(1, 5)],
        "tiny speck dropped (<min_area)": stats["dropped_small_residual"] == 1,
        "A2 keeps only its 20 free px (80 overlap with A)": int(sum(int(m.sum()) for m in ms)) == 320,
        "pixel keep ratio": stats["pixel_keep_ratio"] > 0.0,
    }
    print("\n[ensemble] SELF-TEST")
    ok = True
    for name, cond in checks.items():
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)
    # same merge with near-duplicate suppression: A2 residual is 20/100 = 0.2 < 0.5 -> dropped
    out2, stats2 = merge_frames([(df1, None), (df2, "confidence")],
                                h=hh, w=ww, min_area=20, weights=[1.2, 1.0],
                                max_residual_frac=0.5, verbose=False)
    checks.update({
        "max-residual-frac drops near-duplicate": len(out2) == 3,
        "dropped_duplicate counter": stats2["dropped_duplicate"] == 1,
    })
    for name, cond in list(checks.items()):
        if name.startswith(("max-residual", "dropped_duplicate")):
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
            ok = ok and bool(cond)
    for name in ("dropped_full_overlap", "dropped_small_residual", "dropped_duplicate",
                 "output_rows", "pixel_keep_ratio"):
        print(f"    {name} = {stats[name]}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="merge submissions with score-ordered non-overlap")
    ap.add_argument("--inputs", nargs="+", default=None, help="2+ submission CSV files")
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="per-input weights (default 1.0 each)")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--min-area", type=int, default=20, help="drop residual masks smaller than this")
    ap.add_argument("--max-residual-frac", type=float, default=0.0,
                    help="0 = keep every residual; e.g. 0.5 drops near-duplicates whose free "
                         "fraction is below 50%%")
    ap.add_argument("--score-cols", default=",".join(DEFAULT_SCORE_COLS),
                    help="candidate confidence column names, comma separated")
    ap.add_argument("--stats-json", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.inputs or not args.out:
        ap.error("--inputs and --out are required (or use --selftest)")
    if len(args.inputs) < 2:
        ap.error("give at least two input CSVs")
    score_cols = tuple(s.strip() for s in args.score_cols.split(",") if s.strip())
    merge_files(args.inputs, args.out, weights=args.weights, min_area=args.min_area,
                score_cols=score_cols, max_residual_frac=args.max_residual_frac,
                stats_json=args.stats_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
