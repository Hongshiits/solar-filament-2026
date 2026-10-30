
"""Merge two instance submissions: base mask set + novel instances from the other (IoU<0.5), de-overlapped."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from pycocotools import mask as mask_utils
sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"D:\PROJECT\kaggle\filament-segmentation-2026")
H = W = 2048
DARIUSH = BASE/"submissions/dariush_unet_full_v1.csv"
VOTE15 = BASE/"submissions/candidates/nk_v3_vote15.csv"
OUT = BASE/"submissions/candidates"; OUT.mkdir(parents=True, exist_ok=True)

def load(path):
    df = pd.read_csv(path)
    out = {}
    for fid, rle in zip(df.filament_id.astype(str), df.segmentation_rle.astype(str)):
        stem = fid.rsplit("_", 1)[0]
        m = mask_utils.decode({"counts": rle.encode("ascii"), "size": [H, W]}).astype(np.uint8)
        out.setdefault(stem, []).append(m)
    return out

def rle_of(m):
    c = mask_utils.encode(np.asfortranarray(m))["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else str(c)

def entry(m):
    ys, xs = np.where(m)
    if len(xs) == 0: return None
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    return {"x0": int(x0), "y0": int(y0), "m": m[y0:y1, x0:x1].astype(np.uint8)}

def entry_iou(a, b):
    x0 = max(a["x0"], b["x0"]); y0 = max(a["y0"], b["y0"])
    x1 = min(a["x0"]+a["m"].shape[1], b["x0"]+b["m"].shape[1]); y1 = min(a["y0"]+a["m"].shape[0], b["y0"]+b["m"].shape[0])
    if x1 <= x0 or y1 <= y0: return 0.0
    va = a["m"][y0-a["y0"]:y1-a["y0"], x0-a["x0"]:x1-a["x0"]]
    vb = b["m"][y0-b["y0"]:y1-b["y0"], x0-b["x0"]:x1-b["x0"]]
    inter = int(np.logical_and(va, vb).sum())
    union = int(a["m"].sum()) + int(b["m"].sum()) - inter
    return inter/union if union else 0.0

def subtract(target, other):
    x0 = max(target["x0"], other["x0"]); y0 = max(target["y0"], other["y0"])
    x1 = min(target["x0"]+target["m"].shape[1], other["x0"]+other["m"].shape[1]); y1 = min(target["y0"]+target["m"].shape[0], other["y0"]+other["m"].shape[0])
    if x1 <= x0 or y1 <= y0: return target
    view = target["m"][y0-target["y0"]:y1-target["y0"], x0-target["x0"]:x1-target["x0"]]
    ov = target["m"][y0-target["y0"]:y1-target["y0"], x0-target["x0"]:x1-target["x0"]] & other["m"][y0-other["y0"]:y1-other["y0"], x0-other["x0"]:x1-other["x0"]]
    if not ov.any(): return target
    upd = target["m"].copy()
    upd[y0-target["y0"]:y1-target["y0"], x0-target["x0"]:x1-target["x0"]][ov.astype(bool)] = 0
    ys, xs = np.where(upd)
    if len(xs) == 0: return None
    return {"x0": target["x0"]+int(xs.min()), "y0": target["y0"]+int(ys.min()), "m": np.ascontiguousarray(upd[ys.min():ys.max()+1, xs.min():xs.max()+1], dtype=np.uint8)}

def merge(base_masks, extra_masks, min_area=200):
    """base entries kept whole; extra instance added if IoU<0.5 with all base and residual area>=min_area."""
    base_entries = [entry(m) for m in base_masks]
    base_entries = [e for e in base_entries if e is not None]
    new = []
    for m in extra_masks:
        e = entry(m)
        if e is None: continue
        if any(entry_iou(e, be) >= 0.5 for be in base_entries): continue
        upd = e
        for be in base_entries:
            upd = subtract(upd, be)
            if upd is None: break
        if upd is None: continue
        for ne in new:
            upd = subtract(upd, ne)
            if upd is None: break
        if upd is None or int(upd["m"].sum()) < min_area: continue
        new.append(upd)
    out = []
    for e in base_entries:
        full = np.zeros((H, W), np.uint8)
        full[e["y0"]:e["y0"]+e["m"].shape[0], e["x0"]:e["x0"]+e["m"].shape[1]] = e["m"]
        out.append(full)
    for e in new:
        full = np.zeros((H, W), np.uint8)
        full[e["y0"]:e["y0"]+e["m"].shape[0], e["x0"]:e["x0"]+e["m"].shape[1]] = e["m"]
        out.append(full)
    return out, len(new)

def main():
    dariush = load(DARIUSH)
    vote = load(VOTE15)
    stems = sorted(set(dariush) | set(vote))
    variants = {
        "union_dariush_base": (dariush, vote),   # dariush masks + vote15 extras
        "union_vote15_base": (vote, dariush),    # vote15 masks + dariush extras
    }
    meta = {}
    for name, (base_set, extra_set) in variants.items():
        rows = []
        total_new = 0
        for stem in stems:
            bm = base_set.get(stem, [])
            em = extra_set.get(stem, [])
            merged, n_new = merge(bm, em)
            total_new += n_new
            for i, m in enumerate(merged, 1):
                rows.append((f"{stem}_{i}", rle_of(m)))
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        path = OUT / f"{name}.csv"
        df.to_csv(path, index=False)
        n_img = df.filament_id.str.rsplit("_", n=1).str[0].nunique()
        meta[name] = {"rows": len(df), "images": int(n_img), "added": total_new}
        print(name, meta[name], flush=True)
    json.dump(meta, open(OUT/"union_meta.json", "w"), indent=1)

if __name__ == "__main__":
    main()
