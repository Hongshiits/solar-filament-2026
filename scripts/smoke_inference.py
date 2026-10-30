#!/usr/bin/env python3
"""CPU smoke test of the repository inference path (2 test images by default).

This does NOT reproduce the full submission; it verifies that
  - the weights load from weights/,
  - the pipeline code imports from src/,
  - detector -> crop refiner -> pixel-disjoint assembly -> COCO RLE -> CSV all execute,
  - the produced CSV passes the local validation checks.
Usage:
    python scripts/smoke_inference.py [--limit 2] [--device cpu]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.cloud_pipeline import (  # noqa: E402
    build_refiner_class, entries_for_config, entry_to_full, load_detector, predict_image,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-root", default=None)
    args_cli = ap.parse_args()

    cfg = yaml.safe_load(open(REPO / "configs/inference.yaml", encoding="utf-8"))
    data_root = Path(args_cli.data_root) if args_cli.data_root else REPO / cfg["data_root"]
    test_dir = data_root / cfg["test_dir"]
    images = sorted(test_dir.glob("*.jpeg"))[: args_cli.limit]
    assert images, f"no test images under {test_dir}"

    from argparse import Namespace
    args = Namespace(img_size=int(cfg["detector"]["imgsz"]), ref_crop=int(cfg["refiner"]["size"]))
    detector = load_detector(REPO / cfg["weights"]["detector"], args)
    SeedRefiner = build_refiner_class()
    payload = torch.load(REPO / cfg["weights"]["refiner"], map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    refiner = SeedRefiner(pretrained=False)
    refiner.load_state_dict(state)
    refiner = refiner.to(args_cli.device).eval()

    H = W = int(cfg["output"]["image_size"])
    conf = float(cfg["detector"]["conf"])
    ref = float(cfg["refiner"]["threshold"])
    tta = str(cfg["refiner"]["tta"])
    rows, overlaps, started = [], 0, time.time()
    for path in images:
        refined = predict_image(path, detector, refiner, args_cli.device, conf, args,
                                imgsz=int(cfg["detector"]["imgsz"]),
                                ref_crop=int(cfg["refiner"]["size"]))
        entries = entries_for_config(refined, conf, ref, tta)
        occupied = np.zeros((H, W), dtype=bool)
        for index, entry in enumerate(entries, 1):
            full = entry_to_full(entry)
            if np.any(np.logical_and(full, occupied)):
                overlaps += 1
            occupied |= full
            counts = mask_utils.encode(np.asfortranarray(full.astype(np.uint8)))["counts"]
            rows.append({"filament_id": f"{path.stem}_{index}",
                         "segmentation_rle": counts.decode("ascii") if isinstance(counts, bytes) else str(counts)})

    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "submission_smoke.csv"
        sub.to_csv(out, index=False)
        areas = []
        for rle in sub["segmentation_rle"]:
            mask = mask_utils.decode({"size": [H, W], "counts": rle.encode("ascii")})
            areas.append(int(mask.sum()))
    print(f"[smoke] images={len(images)} rows={len(sub)} overlaps={overlaps} "
          f"area[min/med/max]={min(areas)}/{int(np.median(areas))}/{max(areas)} "
          f"time={time.time() - started:.1f}s device={args_cli.device}")
    assert overlaps == 0 and len(sub) > 0 and min(areas) > 0
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
