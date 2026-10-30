"""One-key CPU pipeline: fuse dariush test probs with semantic multi-fold test probs.

Example (after cloud sem test probs arrive):
    python src/make_fused_submission.py \
        --dariush-probs work/cloud/runs/probs_dump_v1/probs_dariush_test.npz \
        --sem-probs C:/Users/ptcu_/filament_assets/probs/sem_test_3fold.npz \
        --weights 1.0 0.3 --fuse wmean --keep 0.35 0.99 200 0.85 \
        --out submissions/fused_v2.csv

Steps: load dumps -> weighted probability fusion -> KEEP rule -> submission CSV ->
src/validate_submission.py (per-image non-overlap / RLE / ids) -> stats json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import prob_ensemble as E  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dariush-probs", default=str(ROOT / "work" / "cloud" / "runs" / "probs_dump_v1" / "probs_dariush_test.npz"))
    parser.add_argument("--sem-probs", nargs="*", default=None, help="semantic test prob dump(s), uint8 layout a/b")
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                        help="one weight per dump, dariush first (e.g. 1.0 0.3)")
    parser.add_argument("--fuse", default="wmean", choices=["mean", "wmean", "max", "median"])
    parser.add_argument("--keep", nargs=4, type=float, default=[0.35, 0.99, 200, 0.85])
    parser.add_argument("--out", default=str(ROOT / "submissions" / "fused_v2.csv"))
    parser.add_argument("--meta", default=None)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    keep = dict(t_lo=args.keep[0], t_hi=args.keep[1], min_area=int(args.keep[2]), min_meanp=args.keep[3])

    paths = [args.dariush_probs] + list(args.sem_probs or [])
    dumps = []
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        dumps.append(E.load_prob_npz(path))
        log(f"loaded {path} ({len(dumps[-1])} maps)")
    weights = args.weights if args.weights else [1.0] + [0.3] * (len(dumps) - 1)
    if len(weights) != len(dumps):
        raise ValueError(f"weights {weights} != dumps {len(dumps)}")
    common = sorted(set.intersection(*[set(d) for d in dumps]))
    if len(common) < len(dumps[0]):
        log(f"WARNING: dumps cover different stems; using {len(common)} common images (dariush has {len(dumps[0])})")
    if len(dumps) > 1:
        fused = E.fuse_probs([{stem: d[stem] for stem in common} for d in dumps], weights=weights, mode=args.fuse)
    else:
        fused = {stem: dumps[0][stem] for stem in common}
    log(f"fused {len(fused)} images ({args.fuse}, weights={weights})")

    result = E.prob_to_submission(fused, keep=keep, verbose=True)
    df = result[0] if isinstance(result, tuple) else result
    stats = result[1] if isinstance(result, tuple) else {}
    df = df[["filament_id", "segmentation_rle"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_images = int(df["filament_id"].str.rsplit("_", n=1).str[0].nunique()) if len(df) else 0
    log(f"wrote {len(df)} instances / {n_images} images -> {out}")

    meta = {
        "dumps": paths,
        "weights": weights,
        "fuse": args.fuse,
        "keep": keep,
        "rows": len(df),
        "images": n_images,
        "prob_stats": stats,
        "seconds": round(time.time() - started, 1),
        "out": str(out),
    }
    meta_path = Path(args.meta) if args.meta else out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if not args.no_validate:
        code = subprocess.call([sys.executable, str(ROOT / "src" / "validate_submission.py"), str(out)])
        meta["validate_exit"] = int(code)
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        log(f"validate_submission exit {code}")
    log("DONE " + json.dumps({"rows": len(df), "images": n_images}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
