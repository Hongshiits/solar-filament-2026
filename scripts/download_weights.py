#!/usr/bin/env python3
"""The model weights are committed directly in this repository (weights/best_det.pt,
weights/best_ref.pt; each file is below GitHub's 100 MB limit, so no Git LFS or release
download is needed). This script verifies that the files are present.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = ["weights/best_det.pt", "weights/best_ref.pt"]


def main() -> int:
    missing = [w for w in WEIGHTS if not (REPO / w).is_file()]
    if missing:
        print("missing weight files:", missing)
        print("re-clone the repository (weights are committed directly, see weights/README.md)")
        return 1
    for w in WEIGHTS:
        p = REPO / w
        print(f"{w}: {p.stat().st_size / 1e6:.1f} MB - present")
    print("Verify integrity with: python scripts/check_weights.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
