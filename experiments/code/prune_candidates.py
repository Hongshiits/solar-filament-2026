
"""Prune dariush instances using cross-model support (vote15 masks + sem probs), several strengths."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from pycocotools import mask as mask_utils
sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"D:\PROJECT\kaggle\filament-segmentation-2026")
H = W = 2048
DARIUSH = BASE/"submissions/dariush_unet_full_v1.csv"
VOTE15 = BASE/"submissions/candidates/nk_v3_vote15.csv"
SEM = Path(r"C:\Users\ptcu_\filament_assets\probs\sem_test_probs_5fold.npz")
OUT = BASE/"submissions/candidates"

def load_csv(path):
    df = pd.read_csv(path)
    out = {}
    for fid, rle in zip(df.filament_id.astype(str), df.segmentation_rle.astype(str)):
        stem = fid.rsplit("_", 1)[0]
        out.setdefault(stem, []).append(mask_utils.decode({"counts": rle.encode("ascii"), "size": [H, W]}).astype(np.uint8))
    return out

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

def rle_of(m):
    c = mask_utils.encode(np.asfortranarray(m))["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else str(c)

def main():
    dariush = load_csv(DARIUSH)
    vote = load_csv(VOTE15)
    z = np.load(SEM)
    sem = {str(k): z[k].astype(np.float32)/255.0 for k in z.files}
    stems = sorted(dariush)
    variants = {
        "prune_v5_t05": {"vote_iou": 0.5, "sem_meanp": None},
        "prune_v5_t03": {"vote_iou": 0.3, "sem_meanp": None},
        "prune_v5t05_sem035": {"vote_iou": 0.5, "sem_meanp": 0.35},
        "prune_sem035": {"vote_iou": None, "sem_meanp": 0.35},
    }
    stats = {k: 0 for k in variants}
    files = {k: [] for k in variants}
    for stem in stems:
        dm = dariush[stem]
        de = [entry(m) for m in dm]
        ve = [entry(m) for m in vote.get(stem, [])]
        ve = [e for e in ve if e]
        s = sem.get(stem)
        for m, e in zip(dm, de):
            if e is None: continue
            full = np.zeros((H, W), np.uint8); full[e["y0"]:e["y0"]+e["m"].shape[0], e["x0"]:e["x0"]+e["m"].shape[1]] = e["m"]
            max_iou = max((entry_iou(e, o) for o in ve), default=0.0)
            meanp = float(s[full.astype(bool)].mean()) if s is not None else 1.0
            for name, cfg in variants.items():
                keep = True
                if cfg["vote_iou"] is not None and max_iou < cfg["vote_iou"]:
                    keep = False
                if cfg["sem_meanp"] is not None and meanp < cfg["sem_meanp"]:
                    keep = False
                if keep:
                    files[name].append((stem, m))
                else:
                    stats[name] += 1
    meta = {"dariush_rows": sum(len(v) for v in dariush.values())}
    for name, items in files.items():
        by_stem = {}
        for stem, m in items: by_stem.setdefault(stem, []).append(m)
        rows = []
        for stem in sorted(by_stem):
            for i, m in enumerate(by_stem[stem], 1):
                rows.append((f"{stem}_{i}", rle_of(m)))
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        p = OUT / f"{name}.csv"; df.to_csv(p, index=False)
        n_img = df.filament_id.str.rsplit("_", n=1).str[0].nunique()
        meta[name] = {"rows": len(df), "images": int(n_img), "dropped": stats[name]}
        print(name, meta[name], flush=True)
    json.dump(meta, open(OUT/"prune_meta.json", "w"), indent=1)

if __name__ == "__main__":
    main()
