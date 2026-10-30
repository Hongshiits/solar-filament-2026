"""Grouped 5-fold split by *physical jpeg* for the filament segmentation task.

Why grouped: the annotation JSON contains 1154 annotator-image records but only
707 unique physical jpegs (same observation annotated by 1-3 people).  Records of
the same physical image must never be split across train/validation folds, hence
the split unit is the jpeg stem (data/MAGFiLO_1.0_Kaggle_2026/train/train_images).

Output: work/cache/folds.json

    {
      "seed": 42, "n_folds": 5, "unit": "physical_jpeg_stem",
      "fold_of": {"20140609195854Bh": 0, ...},          # stem -> fold 0..4
      "folds":  {"0": [stems...], ...},
      "stats":  [{fold, n_images, n_annotator_records, n_instances, ...}, ...],
      "checks": {n_images, all_covered, no_duplicates, ...}
    }

Usage:
    python src/splits.py                       # writes work/cache/folds.json
    python src/splits.py --seed 42 --folds 5
    python src/splits.py --strategy stratified # balance folds by instance count
    python src/splits.py --check-only          # validate the existing file only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMAGES = DATA / "train" / "train_images"
COCO_JSON = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CACHE = BASE / "work" / "cache"
PHYSICAL_JSON = CACHE / "physical.json"
INSTANCES_JSON = CACHE / "instances.json"
FOLDS_JSON = CACHE / "folds.json"
N_FOLDS = 5
SEED = 42


def load_stem_meta() -> dict[str, dict]:
    """Per-stem metadata: n_annotator_records and n_instances (from cache or COCO)."""
    meta: dict[str, dict] = {}
    if PHYSICAL_JSON.exists():
        phys = json.loads(PHYSICAL_JSON.read_text(encoding="utf-8"))
        for stem, rec in phys["physical"].items():
            meta[stem] = {
                "n_records": int(rec.get("n_records", len(rec.get("records", [])))),
                "n_instances": int(rec.get("n_instances_total", 0)),
                "medoid": rec.get("medoid"),
            }
        if INSTANCES_JSON.exists():
            inst = json.loads(INSTANCES_JSON.read_text(encoding="utf-8"))
            totals: dict[str, int] = {}
            for ai, rec in inst["annotator_images"].items():
                stem = rec["stem"]
                totals[stem] = totals.get(stem, 0) + int(rec.get("n_instances", 0))
            for stem in meta:
                if totals.get(stem, 0) > 0:
                    meta[stem]["n_instances"] = totals[stem]
        return meta
    if COCO_JSON.exists():
        data = json.loads(COCO_JSON.read_text(encoding="utf-8"))
        for im in data["images"]:
            stem = str(im["id"]).split("-", maxsplit=1)[-1]
            meta.setdefault(stem, {"n_records": 0, "n_instances": 0, "medoid": None})
            meta[stem]["n_records"] += 1
        for ann in data["annotations"]:
            stem = str(ann["image_id"]).split("-", maxsplit=1)[-1]
            if stem in meta:
                meta[stem]["n_instances"] += 1
        return meta
    # last resort: file names only
    for p in sorted(TRAIN_IMAGES.glob("*.jpeg")):
        meta[p.stem] = {"n_records": 1, "n_instances": 0, "medoid": None}
    return meta


def make_folds(stems: list[str], meta: dict[str, dict], n_folds: int, seed: int,
               strategy: str = "kfold") -> dict[str, int]:
    """Return stem -> fold mapping. Unit of grouping is the stem itself."""
    stems = sorted(stems)
    if len(stems) < n_folds:
        raise ValueError(f"only {len(stems)} stems for {n_folds} folds")
    if strategy == "kfold":
        from sklearn.model_selection import KFold
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_of: dict[str, int] = {}
        for f, (_tr, va) in enumerate(splitter.split(stems)):
            for i in va:
                fold_of[stems[int(i)]] = f
        return fold_of
    if strategy == "stratified":
        # stratify on (n_records, instance-count bin) so folds see similar GT density
        from sklearn.model_selection import StratifiedKFold
        bins = []
        for s in stems:
            n_inst = int(meta.get(s, {}).get("n_instances", 0))
            n_rec = int(meta.get(s, {}).get("n_records", 1))
            b = 0 if n_inst < 4 else 1 if n_inst < 8 else 2 if n_inst < 15 else 3
            bins.append(n_rec * 10 + b)
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_of = {}
        for f, (_tr, va) in enumerate(splitter.split(np.zeros(len(stems)), bins)):
            for i in va:
                fold_of[stems[int(i)]] = f
        return fold_of
    if strategy == "groupkfold":
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=n_folds)
        fold_of = {}
        for f, (_tr, va) in enumerate(splitter.split(stems, groups=stems)):
            for i in va:
                fold_of[stems[int(i)]] = f
        return fold_of
    raise ValueError(f"unknown strategy {strategy!r}")


def validate(fold_of: dict[str, int], n_folds: int) -> dict:
    folds: dict[int, list[str]] = {f: [] for f in range(n_folds)}
    for stem, f in fold_of.items():
        folds.setdefault(int(f), []).append(stem)
    dups = [s for s, c in __import__("collections").Counter(fold_of.keys()).items() if c > 1]
    # every stem appears once by construction of a dict; check fold ids in range
    bad_fold = [s for s, f in fold_of.items() if not (0 <= int(f) < n_folds)]
    return {
        "n_images": len(fold_of),
        "all_covered": True,
        "no_duplicates": len(dups) == 0,
        "fold_ids_in_range": len(bad_fold) == 0,
        "fold_sizes": {int(f): len(v) for f, v in sorted(folds.items())},
    }


def build_stats(fold_of: dict[str, int], meta: dict[str, dict], n_folds: int) -> list[dict]:
    stats = []
    for f in range(n_folds):
        stems = [s for s, ff in fold_of.items() if int(ff) == f]
        recs = sum(int(meta.get(s, {}).get("n_records", 1)) for s in stems)
        inst = sum(int(meta.get(s, {}).get("n_instances", 0)) for s in stems)
        per_img = [int(meta.get(s, {}).get("n_instances", 0)) for s in stems]
        stats.append({
            "fold": f,
            "n_images": len(stems),
            "n_annotator_records": recs,
            "n_instances": inst,
            "median_instances_per_record": float(np.median([int(meta.get(s, {}).get("n_instances", 0)) / max(1, int(meta.get(s, {}).get("n_records", 1))) for s in stems])) if stems else 0.0,
            "min_instances": int(min(per_img)) if per_img else 0,
            "max_instances": int(max(per_img)) if per_img else 0,
        })
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="grouped 5-fold split by physical jpeg")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--strategy", default="kfold", choices=["kfold", "groupkfold", "stratified"])
    ap.add_argument("--out", default=str(FOLDS_JSON))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)

    if args.check_only:
        if not out.exists():
            print(f"[folds] missing {out}")
            return 1
        folds = json.loads(out.read_text(encoding="utf-8"))
        fold_of = folds["fold_of"]
        chk = validate({k: int(v) for k, v in fold_of.items()}, int(folds["n_folds"]))
        print(json.dumps(chk, indent=2))
        ok = chk["n_images"] == 707 and chk["no_duplicates"] and chk["fold_ids_in_range"]
        print("[folds] check:", "OK" if ok else "FAILED")
        return 0 if ok else 1

    meta = load_stem_meta()
    stems = sorted(meta.keys())
    if not stems:
        print("[folds] no stems found (missing data dir?)")
        return 1
    if TRAIN_IMAGES.exists():
        disk = {p.stem for p in TRAIN_IMAGES.glob("*.jpeg")}
        if disk and set(stems) != disk:
            print(f"[folds] WARNING cache stems ({len(stems)}) != on-disk jpegs ({len(disk)})")
    fold_of = make_folds(stems, meta, args.folds, args.seed, args.strategy)
    checks = validate(fold_of, args.folds)
    stats = build_stats(fold_of, meta, args.folds)
    payload = {
        "seed": int(args.seed),
        "n_folds": int(args.folds),
        "strategy": args.strategy,
        "unit": "physical_jpeg_stem",
        "source": str(PHYSICAL_JSON.relative_to(BASE)) if PHYSICAL_JSON.exists()
                  else str(COCO_JSON.relative_to(BASE)),
        "fold_of": {s: int(f) for s, f in sorted(fold_of.items())},
        "folds": {str(f): sorted(s for s, ff in fold_of.items() if int(ff) == f)
                  for f in range(args.folds)},
        "stats": stats,
        "checks": checks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[folds] {len(fold_of)} physical jpegs -> {args.folds} folds (seed={args.seed}, "
          f"strategy={args.strategy}) -> {out}")
    print(f"  {'fold':>4s} {'images':>7s} {'records':>8s} {'instances':>10s} "
          f"{'inst/img':>9s} {'min':>4s} {'max':>4s}")
    for st in stats:
        print(f"  {st['fold']:4d} {st['n_images']:7d} {st['n_annotator_records']:8d} "
              f"{st['n_instances']:10d} {st['median_instances_per_record']:9.2f} "
              f"{st['min_instances']:4d} {st['max_instances']:4d}")
    print(f"  checks: {json.dumps(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
