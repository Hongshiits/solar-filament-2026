"""Kaggle-kernel entrypoint: semantic U-Net multi-fold TEST probability inference.

Loads every semantic checkpoint found (sem_<fold>_<loss>.pt), runs full-resolution
2048 test inference with overlap-tiled 1024 crops (T4/8GB safe) and flip TTA,
averages the models' probabilities and writes a uint8 prob npz.

    python src/cloud_sem_test_probs.py --out /kaggle/working/sem_test_probs.npz --tile 1024 --tta 2

The output npz is layout (a): {stem: uint8[2048,2048]}, consumable by
src/prob_ensemble.py and src/make_fused_submission.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
INPUT = Path("/kaggle/input")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", nargs="*", default=None, help="explicit semantic checkpoints (best.pt per fold)")
    parser.add_argument("--out", default="/kaggle/working/sem_test_probs.npz")
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--tta", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    ckpts = [Path(p) for p in (args.ckpt or [])]
    if not ckpts:
        roots = [INPUT] if INPUT.exists() else [Path(".")]
        candidates = []
        for root in roots:
            if not root.exists():
                continue
            candidates += sorted(root.glob("**/models/sem/*.pt"))
            candidates += sorted(root.glob("**/sem_*_*.pt"))
            candidates += sorted(root.glob("**/sem_fold*_best.pt"))
        # keep best weights, drop optimizer-bearing last.pt and converted files
        best = [p for p in candidates if p.name.endswith(".pt") and "_last" not in p.name and "best.pt" in p.name or (p.name.startswith("sem_") and p.name.endswith("_bcedice.pt") or p.name.endswith("_bcedice_cldice.pt"))]
        ckpts = []
        for p in best:
            if p not in ckpts:
                ckpts.append(p)
    if not ckpts:
        raise FileNotFoundError("no semantic checkpoints found under /kaggle/input (pass --ckpt)")
    print("semantic checkpoints:", [str(p) for p in ckpts], flush=True)

    test_dir = args.test_dir
    if test_dir is None:
        candidates = []
        for root in ([INPUT] if INPUT.exists() else [Path(".")]):
            if root.exists():
                candidates += sorted(root.glob("**/test_images"))
        if not candidates:
            raise FileNotFoundError("test_images not found")
        test_dir = str(candidates[0])
    print("test dir:", test_dir, flush=True)

    import infer_semantic_test

    cli = [
        "--ckpt", *[str(p) for p in ckpts],
        "--out", args.out,
        "--test-dir", test_dir,
        "--tile", str(args.tile),
        "--overlap", str(args.overlap),
        "--tta", str(args.tta),
        "--limit", str(args.limit),
        "--device", "cuda:0",
    ]
    code = int(infer_semantic_test.main(cli))
    meta = {"ckpts": [str(p) for p in ckpts], "out": args.out, "tile": args.tile, "tta": args.tta}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("sem test probs written:", args.out, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
