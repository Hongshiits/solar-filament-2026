
"""Sparse-label simulation on fold0 val: prune vs union policies (dariush + sem support)."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from pycocotools import mask as mask_utils
BASE = Path(r"D:\PROJECT\kaggle\filament-segmentation-2026")
sys.path.insert(0, str(BASE/"src"))
import pq_metric as P

H = W = 2048
S8 = np.ones((3,3), dtype=bool)
FOLDS = json.load(open(BASE/"work/cache/folds.json", encoding="utf-8"))
GT_JSON = BASE/"data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
KEEP = dict(t_lo=0.35, t_hi=0.99, min_area=200, min_meanp=0.85)
R = 0.77

def instances_from_probs(p):
    tl = int(round(KEEP["t_lo"]*255)); th = int(round(KEEP["t_hi"]*255))
    lab, n = ndi.label(p > tl, structure=S8)
    if n == 0: return []
    lf = lab.ravel(); nz = np.flatnonzero(lf)
    lv, pv = lf[nz], p.ravel()[nz]
    area = np.bincount(lv, minlength=n+1)[1:].astype(np.int64)
    psum = np.bincount(lv, weights=pv, minlength=n+1)[1:]
    pmax = np.zeros(n+1, np.uint8); np.maximum.at(pmax, lv, pv)
    meanp = psum/np.maximum(area,1)/255.0
    ok = (pmax[1:] > th) & (area >= KEEP["min_area"]) & (meanp >= KEEP["min_meanp"])
    labels = np.flatnonzero(ok)+1
    objs = ndi.find_objects(lab)
    out = []
    for c in labels:
        sl = objs[c-1]
        m = np.zeros((H, W), np.uint8); m[sl] = (lab[sl] == c)
        out.append(m)
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

def subtract(target, other):
    x0 = max(target["x0"], other["x0"]); y0 = max(target["y0"], other["y0"])
    x1 = min(target["x0"]+target["m"].shape[1], other["x0"]+other["m"].shape[1]); y1 = min(target["y0"]+target["m"].shape[0], other["y0"]+other["m"].shape[0])
    if x1 <= x0 or y1 <= y0: return target
    ov = target["m"][y0-target["y0"]:y1-target["y0"], x0-target["x0"]:x1-target["x0"]] & other["m"][y0-other["y0"]:y1-other["y0"], x0-other["x0"]:x1-other["x0"]]
    if not ov.any(): return target
    upd = target["m"].copy()
    upd[y0-target["y0"]:y1-target["y0"], x0-target["x0"]:x1-target["x0"]][ov.astype(bool)] = 0
    ys, xs = np.where(upd)
    if len(xs) == 0: return None
    return {"x0": target["x0"]+int(xs.min()), "y0": target["y0"]+int(ys.min()), "m": np.ascontiguousarray(upd[ys.min():ys.max()+1, xs.min():xs.max()+1], dtype=np.uint8)}

def evaluate(gt_df, rows):
    pred = pd.DataFrame(rows, columns=["filament_id","segmentation_rle"])
    res = P.evaluate(gt_df, pred)
    return res

def main():
    t0 = time.time()
    z = np.load(BASE/"work/dariush_probs/probs_dariush_fold0.npz")
    dprobs = {str(s): z["probs"][i] for i, s in enumerate(z["stems"])}
    zs = np.load(BASE/"work/models/sem/sem_0_bcedice_val_probs_best.npz")
    sprobs = {str(k): zs[k] for k in zs.files}
    stems = sorted(FOLDS["folds"]["0"])
    gt_all = P.gt_df_from_coco(GT_JSON)
    gt = P.gt_df_for_stems(gt_all, stems)
    print("gt rows:", len(gt), flush=True)
    # build instances & sem support
    A = {}; B = {}; C = {}  # candidate instance dicts
    n_inst = 0; n_pruned = 0; n_added = 0
    for stem in stems:
        d_inst = instances_from_probs(dprobs[stem])
        s_inst = instances_from_probs(sprobs[stem])
        sem = sprobs[stem].astype(np.float32)/255.0
        d_entries = [entry(m) for m in d_inst]
        s_entries = [entry(m) for m in s_inst]
        s_entries = [e for e in s_entries if e]
        rowsA, rowsB = [], []
        for m, e in zip(d_inst, d_entries):
            full = np.zeros((H,W), np.uint8); full[e["y0"]:e["y0"]+e["m"].shape[0], e["x0"]:e["x0"]+e["m"].shape[1]] = e["m"]
            vals = sem[full.astype(bool)]
            mean_sem = float(vals.mean()) if len(vals) else 0.0
            rowsA.append(rle_of(full)); n_inst += 1
            if mean_sem >= 0.30:
                rowsB.append(rle_of(full))
            else:
                n_pruned += 1
        # C = A + sem extras not overlapping dariush
        rowsC = list(rowsA)
        for m, e in zip(s_inst, s_entries):
            if any(entry_iou(e, de) >= 0.5 for de in d_entries if de):
                continue
            upd = e
            for de in d_entries:
                if de: upd = subtract(upd, de)
                if upd is None: break
            if upd is None or int(upd["m"].sum()) < 200: continue
            full = np.zeros((H,W), np.uint8); full[upd["y0"]:upd["y0"]+upd["m"].shape[0], upd["x0"]:upd["x0"]+upd["m"].shape[1]] = upd["m"]
            rowsC.append(rle_of(full)); n_added += 1
        A[stem] = rowsA; B[stem] = rowsB; C[stem] = rowsC
    print(f"instances={n_inst} pruned={n_pruned} added={n_added} ({time.time()-t0:.0f}s)", flush=True)

    def to_rows(d):
        rows = []
        for stem in stems:
            for i, r in enumerate(d[stem], 1):
                rows.append((f"{stem}_{i}", r))
        return rows
    candidates = {"A_dariush_full": to_rows(A), "B_dariush_pruned": to_rows(B), "C_dariush_plus_sem": to_rows(C)}
    # regimes: full GT + sparse GT (random drop 23% per annotator-image)
    results = {k: {} for k in candidates}
    base_res = {k: evaluate(gt, v) for k, v in candidates.items()}
    for k, v in base_res.items():
        results[k]["full"] = {kk: v[kk] for kk in ("pq","sq","rq","tp","fp","fn","n_pred")}
        print(f"{k}: full GT PQ={v['pq']:.4f} TP={v['tp']} FP={v['fp']} FN={v['fn']}", flush=True)
    rng = np.random.default_rng(0)
    # sparse: drop 23% of annotation rows per annotator-image
    keys = ["filament_id"]
    stem_col = gt["filament_id"].astype(str).str.split("_", n=1).str[0]
    sparse_scores = {k: [] for k in candidates}
    for seed in range(8):
        rng = np.random.default_rng(seed)
        keep_mask = rng.random(len(gt)) >= 0.23
        gt_sp = gt[keep_mask].reset_index(drop=True)
        for k, v in candidates.items():
            res = evaluate(gt_sp, v)
            sparse_scores[k].append(float(res["pq"]))
    for k in candidates:
        results[k]["sparse_mean"] = float(np.mean(sparse_scores[k]))
        results[k]["sparse_std"] = float(np.std(sparse_scores[k]))
        print(f"{k}: sparse(0.77) PQ={results[k]['sparse_mean']:.4f} ± {results[k]['sparse_std']:.4f}", flush=True)
    json.dump({"n_instances": n_inst, "n_pruned": n_pruned, "n_added": n_added, "results": results, "r": R},
              open(BASE/"work/analysis/ensemble/sparse_policy_sim.json","w"), indent=1, default=float)
    print("done", round(time.time()-t0,1), "s")

if __name__ == "__main__":
    main()
