#!/usr/bin/env python3
"""Verify that the model weights required for reproduction are present and intact.

The evaluation requires that the notebook loads the trained weights stored in the
repository, so this is the first thing to run after cloning.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO / "weights"

# filename -> (expected md5, expected sha256, minimum size in bytes)
EXPECTED = {
    "best_det.pt": (
        "fd91ebd028e0ab58d7ff726013104c92",
        "fd3f534f770b7027ba900bd4d96b3559e5888b612868ec86eaf3ae16b8396814",
        15_000_000,
    ),
    "best_ref.pt": (
        "678a319cb4bd58d507ea89ea3b1d3a7f",
        "581280342f1ee04ea0d0cef6f8fceb449264392e5c9fe737e03f74894e618409",
        50_000_000,
    ),
}


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    failures = 0
    for name, (expected_md5, expected_sha256, min_size) in EXPECTED.items():
        path = WEIGHTS_DIR / name
        if not path.is_file():
            print(f"[MISSING] {path}")
            print("          see weights/README.md for how to obtain it")
            failures += 1
            continue
        size = path.stat().st_size
        md5 = digest(path, "md5")
        sha256 = digest(path, "sha256")
        status = "OK"
        if size < min_size:
            status = "SUSPICIOUS_SIZE"
        elif md5 != expected_md5 or sha256 != expected_sha256:
            status = "HASH_MISMATCH"
        if status != "OK":
            failures += 1
        print(f"[{status}] {name}: {size / 1e6:.1f} MB  md5={md5}  sha256={sha256}")
    if failures:
        print(f"\n{failures} weight file(s) not ready -> the notebook cannot reproduce the submission.")
        return 1
    print("\nAll weights OK. Continue with notebooks/final_pipeline.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
