
"""Precision-oriented candidates: intersection (instances supported by both 0.39 solutions) + high-min-area union."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from pycocotools import mask as mask_utils
sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"D:\PROJECT\kaggle\filament-segmentation-2026")
H = W = 2048
DARIUSH = BASE/"submissions/dariush_unet_full_v1.csv"
VOTE15 = BASE/"submissions/candidates/nk_v3_vote15.csv"
OUT = BASE/"submissions/candidates"

def load(path):
    df = pd.read_csv(path)
    out = {}
    for fid, rle in zip(df.filament_id.astype(str), df.segmentation_rle.astype(str)):
        stem = fid.rsplit("_", 1)[0]
        out.setdefault(stem, []).append(mask_utils.decode({"counts": rle.encode("ascii"), "size": [H, W]}).astype(np.uint8))
    return out

def rle_of(m):
    c = mask_utils.encode(np.asfortranarray(m))["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else str(c)

def entry(m):
    ys, xs = np.where(m)
    if len(xs) == 0: return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1
    return {"x0": x0, "y0": y0, "m": m[y0:y1, x0:x1].astype(np.uint8)}

def entry_iou(a, b):
    x0 = max(a["x0"], b["x0"]); y0 = max(a["y0"], b["y0"])
    x1 = min(a["x0"]+a["m"].shape[1], b["x0"]+b["m"].shape[1]); y1 = min(a["y0"]+a["m"].shape[0], b["y0"]+b["m"].shape[0])
    if x1 <= x0 or y1 <= y0: return 0.0
    va = a["m"][y0-a["y0"]:y1-a["y0"], x0-a["x0"]:x1-a["x0"]]
    vb = b["m"][y0-b["y0"]:y1-b["y0"], x0-b["x0"]:x1-b["x0"]]
    inter = int(np.logical_and(va, vb).sum())
    union = int(a["m"].sum()) + int(b["m"].sum()) - inter
    return inter/union if union else 0.0

def write(rows, name):
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False)
    n_img = df.filament_id.str.rsplit("_", n=1).str[0].nunique() if len(df) else 0
    print(name, {"rows": len(df), "images": int(n_img)}, flush=True)
    return {"rows": len(df), "images": int(n_img)}

def main():
    dariush = load(DARIUSH); vote = load(VOTE15)
    stems = sorted(set(dariush) | set(vote))
    meta = {}
    # A) intersection: dariush instances with a vote15 match IoU>=0.5 (keep dariush mask)
    rows_a = []
    rows_b = []
    for stem in stems:
        dm = dariush.get(stem, []); vm = vote.get(stem, [])
        de = [entry(m) for m in dm]; de = [e for e in de if e]
        ve = [entry(m) for m in vm]; ve = [e for e in ve if e]
        ka = 0
        for m, e in zip(dm, de):
            if any(entry_iou(e, o) >= 0.5 for o in ve):
                ka += 1
                rows_a.append((f"{stem}_{ka}", rle_of(m)))
        kb = 0
        for m, e in zip(vm, ve):
            if any(entry_iou(e, o) >= 0.5 for o in de):
                kb += 1
                rows_b.append((f"{stem}_{kb}", rle_of(m)))
    meta["inter_dariush"] = write(rows_a, "inter_dariush")
    meta["inter_vote15"] = write(rows_b, "inter_vote15")
    json.dump(meta, open(OUT/"intersection_meta.json", "w"), indent=1)

if __name__ == "__main__":
    main()