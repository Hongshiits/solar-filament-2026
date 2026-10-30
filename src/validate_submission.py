
"""提交文件校验（Lead 统一使用）。
用法: python src/validate_submission.py submissions/xxx.csv
检查: 列名、filament_id 格式、image_id 覆盖、RLE 可解码、面积>0、无重复、CSV 可读。
"""
from __future__ import annotations
import sys, os, re, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "MAGFiLO_1.0_Kaggle_2026"
H = W = 2048

def test_stems() -> list[str]:
    d = DATA / "test" / "test_images"
    return sorted(p.stem for p in d.glob("*.jpeg"))

def main(csv_path: str) -> int:
    stems = set(test_stems())
    path = Path(csv_path)
    print(f"[validate] {path}  size={path.stat().st_size/1e6:.2f} MB")
    df = pd.read_csv(path, dtype=str)
    errs: list[str] = []
    expect = ["filament_id", "segmentation_rle"]
    if list(df.columns[:2]) != expect:
        errs.append(f"columns={list(df.columns)[:4]} expected {expect}")
    if len(df) == 0:
        errs.append("empty submission")
    # ids
    ids = df["filament_id"].astype(str)
    pat = re.compile(r"^([A-Za-z0-9]+)_(\d+)$")
    bad_ids = [x for x in ids if not pat.match(x)]
    if bad_ids: errs.append(f"bad filament_id x{len(bad_ids)} e.g. {bad_ids[:3]}")
    dups = ids[ids.duplicated()].unique()
    if len(dups): errs.append(f"duplicate filament_id x{len(dups)} e.g. {list(dups[:3])}")
    img_ids = ids.str.extract(pat.pattern)[0]
    unknown = sorted(set(img_ids.dropna()) - stems)
    if unknown: errs.append(f"unknown image_ids x{len(unknown)} e.g. {unknown[:3]}")
    covered = len(set(img_ids.dropna()) & stems)
    # rle decode
    n_empty = 0; n_zero = 0; bad_decode = 0; areas = []
    for i, rle in enumerate(df["segmentation_rle"].astype(str)):
        if not rle or rle == "nan":
            n_empty += 1; continue
        try:
            m = mask_utils.decode({"size": [H, W], "counts": rle})
            a = int(m.sum())
        except Exception:
            bad_decode += 1
            if bad_decode <= 3:
                print(f"  [decode-fail] row {i} id={ids.iloc[i]} rle={rle[:60]}")
            continue
        if a == 0: n_zero += 1
        areas.append(a)
    if n_empty: errs.append(f"empty RLE strings x{n_empty}")
    if bad_decode: errs.append(f"undecodable RLE x{bad_decode}")
    if n_zero: errs.append(f"zero-area masks x{n_zero}")
    # ---- hard constraint: per-image pixel non-overlap (else SubmissionStatus.ERROR) ----
    overlaps = []
    if not (n_empty or bad_decode):
        by_img: dict[str, list[int]] = {}
        for i, img in enumerate(img_ids):
            if isinstance(img, str):
                by_img.setdefault(img, []).append(i)
        for img, idxs in by_img.items():
            acc = np.zeros((H, W), dtype=np.uint16)
            for i in idxs:
                acc += mask_utils.decode({"size": [H, W], "counts": df["segmentation_rle"].astype(str).iloc[i]}).astype(np.uint16)
            n_over = int((acc > 1).sum())
            if n_over:
                overlaps.append((img, n_over, len(idxs)))
        if overlaps:
            print(f"[validate] images with overlapping predictions: {len(overlaps)} e.g. {overlaps[:5]}")
            errs.append(f"OVERLAPPING masks in {len(overlaps)} images (would be SubmissionStatus.ERROR)")
    if areas:
        arr = np.array(areas)
        print(f"[validate] rows={len(df)} images_covered={covered}/{len(stems)} area[min/med/max]={arr.min()}/{int(np.median(arr))}/{arr.max()} total_area={arr.sum()/1e6:.3f}Mpx")
    else:
        print(f"[validate] rows={len(df)} images_covered={covered}/{len(stems)} (no decodable masks)")
    if errs:
        print("[validate] FAILED:")
        for e in errs: print("  -", e)
        return 1
    print("[validate] OK")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
