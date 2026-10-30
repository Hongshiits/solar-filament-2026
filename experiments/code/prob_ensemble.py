"""Probability-map ensemble + KEEP-rule post-processing for the filament task.

Two responsibilities:

1. Fuse several probability dicts (stem -> uint8 probability map, 2048x2048,
   as produced by the dariush U-Net TTA dump) into one probability dict
   (mean / weighted mean / median / max), then apply the official KEEP rule to
   turn it into instance masks + submission RLEs.

2. Apply the KEEP rule directly to probability maps (single source or fused):
       regions of p > t_lo (8-connected) kept if
         max p > t_hi  and  area >= min_area px  and  mean p >= min_meanp
   Thresholds are compared on the uint8 maps as round(t * 255), exactly like the
   dariush receipt (solar_unet_receipt.json, keep_rule).

Probability dict conventions
----------------------------
npz files are accepted in two layouts:
  (a) one entry per image: keys are the stems, values are 2-D uint8 arrays;
  (b) packed: key "stems" (array of str) plus "probs" (N, H, W) uint8, or keys
      "prob_0", "prob_1", ... in the same order as "stems".
A sidecar JSON with the same basename (.json) is optional; if it contains
"stems" (and optionally "height"/"width") it is used to name arrays of layout (b).
Use save_prob_npz() to write the canonical layouts.

CLI
---
    # fuse several dumps and produce a submission
    python src/prob_ensemble.py --prob-npz a.npz b.npz --weights 1.0 0.5 \
        --fuse mean --keep 0.35 0.99 200 0.85 --out submissions/dariush_fused.csv
    # score on one grouped-CV fold (val stems of work/cache/folds.json)
    python src/prob_ensemble.py --prob-npz val.npz --keep 0.35 0.99 200 0.85 \
        --eval-fold 0
    python src/prob_ensemble.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
import pq_metric as P  # noqa: E402

DEFAULT_KEEP = dict(t_lo=0.35, t_hi=0.99, min_area=200, min_meanp=0.85, connectivity=8)
H = W = P.H


# ======================================================================================
# 1. probability IO
# ======================================================================================
def _as_stem_array(z, sidecar: dict | None) -> dict[str, np.ndarray]:
    keys = list(z.keys())
    if "stems" in keys:
        stems = [str(s) for s in np.asarray(z["stems"]).reshape(-1).tolist()]
        if "probs" in keys:
            probs = z["probs"]
        else:
            probs = np.stack([z[f"prob_{i}"] for i in range(len(stems))])
        if probs.shape[0] != len(stems):
            raise ValueError(f"probs has {probs.shape[0]} maps but {len(stems)} stems")
        return {s: probs[i] for i, s in enumerate(stems)}
    if sidecar and "stems" in sidecar:
        stems = [str(s) for s in sidecar["stems"]]
        if "probs" in keys:
            probs = z["probs"]
            if probs.shape[0] != len(stems):
                raise ValueError(f"probs has {probs.shape[0]} maps but {len(stems)} stems")
            return {s: probs[i] for i, s in enumerate(stems)}
        if all(k.startswith("prob_") for k in keys):
            keys = sorted(keys, key=lambda k: int(k.split("_")[1]))
            if len(keys) != len(stems):
                raise ValueError(f"{len(keys)} prob_* arrays but {len(stems)} stems in sidecar")
            return {s: z[k] for s, k in zip(stems, keys)}
    out = {}
    for k in keys:
        v = z[k]
        if v.ndim == 2:
            out[str(k)] = v
    if not out:
        raise ValueError(f"cannot interpret npz with keys {keys[:6]} (need stem->2D maps "
                         f"or 'stems'+'probs')")
    return out


def load_prob_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a probability dump -> {stem: uint8 (H, W) array}."""
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        sidecar = None
        side = p.with_suffix(".json")
        if side.exists():
            try:
                sidecar = json.loads(side.read_text(encoding="utf-8"))
            except Exception:
                sidecar = None
        out = _as_stem_array(z, sidecar)
    for stem, arr in out.items():
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            scale = 255.0 if float(arr.max()) <= 1.0 else 1.0
            arr = np.clip(np.rint(arr.astype(np.float32) * scale), 0, 255).astype(np.uint8)
            out[stem] = arr
    return out


def save_prob_npz(prob: dict[str, np.ndarray], path: str | Path, packed: bool = False) -> None:
    """Save {stem: uint8 (H, W)} maps; packed=True uses stems/probs arrays."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stems = sorted(prob)
    if packed:
        np.savez_compressed(p, stems=np.asarray(stems), probs=np.stack([prob[s] for s in stems]))
    else:
        np.savez_compressed(p, **{s: prob[s] for s in stems})


# ======================================================================================
# 2. fusion
# ======================================================================================
def fuse_probs(prob_dicts, weights=None, mode: str = "mean",
               h: int | None = None, w: int | None = None) -> dict[str, np.ndarray]:
    """Fuse several {stem: uint8 map} dicts into one (mean / wmean / median / max).

    Missing stems in one source are skipped for that stem (fuse over available
    sources only).  Output values are uint8 in [0, 255].
    """
    if not prob_dicts:
        raise ValueError("no probability dicts given")
    if weights is None:
        weights = [1.0] * len(prob_dicts)
    if len(weights) != len(prob_dicts):
        raise ValueError("weights length != number of dicts")
    stems = sorted(set().union(*[set(d) for d in prob_dicts]))
    out = {}
    for stem in stems:
        maps = [d[stem].astype(np.float32) for d in prob_dicts if stem in d]
        wts = [float(wt) for d, wt in zip(prob_dicts, weights) if stem in d]
        shape = maps[0].shape
        if mode == "mean":
            acc = np.zeros(shape, dtype=np.float32)
            for m in maps:
                acc += m
            fused = acc / len(maps)
        elif mode == "wmean":
            wsum = sum(wts)
            acc = np.zeros(shape, dtype=np.float32)
            for m, wt in zip(maps, wts):
                acc += m * wt
            fused = acc / wsum
        elif mode == "median":
            fused = np.median(np.stack(maps), axis=0)
        elif mode == "max":
            fused = np.max(np.stack(maps), axis=0)
        else:
            raise ValueError(f"unknown fuse mode {mode!r}")
        out[stem] = np.clip(np.rint(fused), 0, 255).astype(np.uint8)
    return out


# ======================================================================================
# 3. KEEP rule: uint8 probability map -> instance masks
# ======================================================================================
def _thr_u8(t: float) -> int:
    """KEEP-rule threshold on the uint8 map, as round(t * 255)."""
    return int(round(float(t) * 255.0))


def components_with_stats(prob_u8: np.ndarray, t_lo: float, area_floor: int = 20,
                         connectivity: int = 8):
    """All connected components of (p > t_lo) with per-component stats.

    Returns (lab, cstats, ids, areas, maxp, meanp) where arrays are aligned with
    ids (component label ids in cv2.connectedComponentsWithStats order, area
    filter applied).  Returns None when no component passes area_floor.
    """
    import cv2
    prob = np.ascontiguousarray(prob_u8)
    seed = (prob > _thr_u8(t_lo)).astype(np.uint8)
    n, lab, cstats, _ = cv2.connectedComponentsWithStats(seed, connectivity=int(connectivity))
    if n <= 1:
        return None
    areas_all = cstats[1:, cv2.CC_STAT_AREA]
    ids = np.flatnonzero(areas_all >= int(area_floor)) + 1
    if ids.size == 0:
        return None
    areas = np.empty(ids.size, dtype=np.int64)
    maxp = np.empty(ids.size, dtype=np.uint8)
    meanp = np.empty(ids.size, dtype=np.float64)
    for j, cid in enumerate(ids):
        x = int(cstats[cid, cv2.CC_STAT_LEFT]); y = int(cstats[cid, cv2.CC_STAT_TOP])
        w = int(cstats[cid, cv2.CC_STAT_WIDTH]); h = int(cstats[cid, cv2.CC_STAT_HEIGHT])
        sub = prob[y:y + h, x:x + w]
        sel = lab[y:y + h, x:x + w] == cid
        vals = sub[sel]
        areas[j] = int(cvals_area := vals.size)
        maxp[j] = int(vals.max()) if vals.size else 0
        meanp[j] = float(vals.mean()) if vals.size else 0.0
    return lab, cstats, ids, areas, maxp, meanp


def keep_rule_components(prob_u8: np.ndarray, keep: dict | None = None):
    """Fast KEEP rule pass: returns (lab, stats, kept_ids) or None if no regions.

    Uses cv2.connectedComponentsWithStats (C, gives area+bbox) and evaluates the
    max/mean probability only on components that already pass the area filter,
    inside their bounding box.  Typically ~5-20 ms per 2048x2048 map instead of
    ~370 ms with scipy.ndimage reductions.
    """
    k = dict(DEFAULT_KEEP)
    if keep:
        k.update(keep)
    out = components_with_stats(prob_u8, k["t_lo"], area_floor=int(k["min_area"]),
                                connectivity=int(k.get("connectivity", 8)))
    if out is None:
        return None
    lab, cstats, ids, _areas, maxp, meanp = out
    thr_hi = _thr_u8(k["t_hi"])
    thr_mean = float(k["min_meanp"]) * 255.0
    ok = (maxp > thr_hi) & (meanp >= thr_mean)
    kept = [int(cid) for cid in ids[ok]]
    return lab, cstats, kept


def keep_rule_instances(prob_u8: np.ndarray, keep: dict | None = None,
                        return_labels: bool = False):
    """Apply the official KEEP rule to one uint8 probability map.

    Returns a list of disjoint bool instance masks (or a filtered uint16 label
    map if return_labels).  Regions with area < min_area are dropped.
    """
    prob = np.ascontiguousarray(prob_u8)
    comp = keep_rule_components(prob, keep)
    if comp is None:
        empty = np.zeros(prob.shape, dtype=np.uint16)
        return (empty, empty) if return_labels else []
    lab, _stats, kept = comp
    if not kept:
        empty = np.zeros(prob.shape, dtype=np.uint16)
        return (empty, empty) if return_labels else []
    lut = np.zeros(int(lab.max()) + 1, dtype=np.uint16)
    lut[np.asarray(kept, dtype=np.int64)] = np.arange(1, len(kept) + 1, dtype=np.uint16)
    out_lab = lut[lab]
    if return_labels:
        return out_lab, out_lab
    return [out_lab == i for i in range(1, len(kept) + 1)]


def rles_from_crop_label(label_crop: np.ndarray, r0: int, c0: int, H: int, W: int) -> list[str]:
    """Convert a cropped uint16 instance label map to full-image RLE strings.

    Correct for crops of any size: every foreground pixel is mapped to its
    absolute column-major position in the full image and then contiguous runs are
    rebuilt (a run in crop-column-major order is NOT necessarily contiguous in the
    full image when it wraps across a crop column boundary, so we work from pixel
    positions, not from crop runs).
    """
    L = np.ascontiguousarray(label_crop)
    n_inst = int(L.max()) if L.size else 0
    if n_inst == 0:
        return []
    ch, cw = L.shape
    Lf = L.reshape(-1, order="F")
    pos = np.flatnonzero(Lf)                # local indices of all foreground pixels
    if pos.size == 0:
        return []
    vals = Lf[pos].astype(np.int64)
    col = pos // ch
    row = pos % ch
    abs_pos = (int(r0) + row) + (int(c0) + col) * int(W)   # monotone within an instance
    hw = int(H) * int(W)
    out: list[str] = []
    for i in range(1, n_inst + 1):
        sel = vals == i
        if not sel.any():
            continue
        p = abs_pos[sel]
        brk = np.flatnonzero(np.diff(p) != 1) + 1
        starts = np.concatenate((p[:1], p[brk]))
        ends = np.concatenate((p[brk - 1] + 1, p[-1:] + 1))
        lengths = ends - starts
        k = starts.size
        counts = np.empty(2 * k + 1, dtype=np.int64)
        counts[1::2] = lengths
        counts[0] = starts[0]
        if k > 1:
            counts[2:2 * k:2] = starts[1:] - (starts[:-1] + lengths[:-1])
        counts[-1] = hw - (starts[-1] + lengths[-1])
        out.append(P.compressed_from_counts(counts))
    return out


def instance_masks_to_rles(masks, h: int = H, w: int = W) -> list[str]:
    """Encode disjoint bool masks (same shape as (h, w)) as compressed COCO RLEs."""
    return [P.encode_mask(np.ascontiguousarray(m, dtype=np.uint8)) for m in masks]


def prob_to_submission(prob: dict[str, np.ndarray], keep: dict | None = None,
                       min_area: int | None = None, h: int | None = None,
                       w: int | None = None, limit: int | None = None,
                       verbose: bool = True):
    """Apply KEEP rule to every probability map -> submission DataFrame + stats."""
    import pandas as pd
    k = dict(DEFAULT_KEEP)
    if keep:
        k.update(keep)
    if min_area is not None:
        k["min_area"] = int(min_area)
    stems = sorted(prob)
    if h is None or w is None:
        h, w = next(iter(prob.values())).shape
    if limit is not None:
        stems = stems[:int(limit)]
    rows = []
    stats = {"n_images": 0, "n_images_with_pred": 0, "n_instances": 0,
             "height": int(h), "width": int(w),
             "area_total": 0, "per_image_instances": []}
    import cv2
    t0 = time.time()
    for n_done, stem in enumerate(stems, 1):
        pmap = prob[stem]
        if pmap.shape != (h, w):
            raise ValueError(f"{stem}: prob shape {pmap.shape} != ({h},{w})")
        comp = keep_rule_components(pmap, k)
        stats["n_images"] += 1
        if comp is None or not comp[2]:
            stats["per_image_instances"].append(0)
            continue
        lab, cstats, kept = comp
        stats["n_images_with_pred"] += 1
        stats["n_instances"] += len(kept)
        stats["per_image_instances"].append(len(kept))
        # union bbox of kept components -> cropped label -> full-image RLEs
        ys, xs, ye, xe = [], [], [], []
        for cid in kept:
            x = int(cstats[cid, cv2.CC_STAT_LEFT]); y = int(cstats[cid, cv2.CC_STAT_TOP])
            wd = int(cstats[cid, cv2.CC_STAT_WIDTH]); ht = int(cstats[cid, cv2.CC_STAT_HEIGHT])
            ys.append(y); xs.append(x); ye.append(y + ht); xe.append(x + wd)
        r0, r1 = min(ys), max(ye)
        c0, c1 = min(xs), max(xe)
        lab_crop = lab[r0:r1, c0:c1].copy()
        lut = np.zeros(int(lab.max()) + 1, dtype=np.uint16)
        lut[np.asarray(kept, dtype=np.int64)] = np.arange(1, len(kept) + 1, dtype=np.uint16)
        lab_crop = lut[lab_crop]
        rles = rles_from_crop_label(lab_crop, r0, c0, h, w)
        stats["area_total"] += int(sum(int(cstats[cid, 4]) for cid in kept))
        for i, rle in enumerate(rles, start=1):
            rows.append({"filament_id": f"{stem}_{i}", "segmentation_rle": rle})
        if verbose and n_done % 50 == 0:
            print(f"  [prob_ensemble] {n_done}/{len(stems)} images ({time.time() - t0:.1f}s)",
                  flush=True)
    import pandas as pd
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    stats["rows"] = int(len(df))
    stats["runtime_seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(f"[prob_ensemble] {stats['n_images_with_pred']}/{stats['n_images']} images with "
              f"prediction, {stats['rows']} instances, {stats['runtime_seconds']}s")
    return df, stats


# ======================================================================================
# 3b. multi-scale / multi-source loading helpers
# ======================================================================================
import re as _re


def _scale_name(p) -> str:
    p = Path(p)
    m = _re.search(r"s(\d{2,3})", p.stem)
    if m:
        return f"s{int(m.group(1))}"
    base = p.stem
    for prefix in ("probs_", "prob_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
    return base


def load_multiscale(paths_or_dir) -> dict[str, dict[str, np.ndarray]]:
    """Load multi-scale probability dumps -> {scale_name: {stem: uint8 map}}.

    Accepted layouts:
      * a directory containing one npz per scale (scale name inferred from the
        file name, e.g. probs_s085.npz -> s85);
      * a list of such npz paths;
      * one npz with keys like probs_s085/probs_s100/probs_s115 (+ stems);
      * one npz with probs of shape (n_scales, N, H, W) and optional scales.
    """
    if isinstance(paths_or_dir, (str, Path)):
        p = Path(paths_or_dir)
        if p.is_dir():
            files = sorted(p.glob("*.npz"))
            if not files:
                raise FileNotFoundError(f"no npz files in {p}")
            return {_scale_name(f): load_prob_npz(f) for f in files}
        paths = [p]
    else:
        paths = [Path(x) for x in paths_or_dir]
    if len(paths) > 1:
        return {_scale_name(f): load_prob_npz(f) for f in paths}
    p = paths[0]
    with np.load(p, allow_pickle=False) as z:
        keys = list(z.keys())
        if "probs" in keys and z["probs"].ndim == 4:
            probs = z["probs"]
            if "stems" in keys:
                stems = [str(s) for s in np.asarray(z["stems"]).reshape(-1).tolist()]
            else:
                stems = [f"img{i}" for i in range(probs.shape[1])]
            if "scales" in keys:
                scales = [str(s) for s in np.asarray(z["scales"]).reshape(-1).tolist()]
            else:
                scales = [f"s{i}" for i in range(probs.shape[0])]
            if probs.shape[1] != len(stems):
                raise ValueError(f"probs has {probs.shape[1]} images but {len(stems)} stems")
            return {sc: {stem: probs[i, j] for j, stem in enumerate(stems)}
                    for i, sc in enumerate(scales)}
        multi_keys = [k for k in keys if k.startswith("probs_")]
        if multi_keys and "stems" in keys:
            stems = [str(s) for s in np.asarray(z["stems"]).reshape(-1).tolist()]
            out = {}
            for k in multi_keys:
                arr = z[k]
                out[k[len("probs_"):]] = {stem: arr[j] for j, stem in enumerate(stems)}
            return out
    return {"default": load_prob_npz(p)}


def sparse_pq(res: dict, r: float = 0.77) -> float:
    """Expected PQ when test GT is annotated sparsely with rate r (dariush formula).

    PQ'(r) = r*sum_iou / ( r*tp + 0.5*fp + 0.5*(1-r)*tp + 0.5*r*fn ), sum_iou = sq*tp
    """
    tp = float(res["tp"]); fp = float(res["fp"]); fn = float(res["fn"])
    sum_iou = float(res["sq"]) * tp
    denom = r * tp + 0.5 * fp + 0.5 * (1.0 - r) * tp + 0.5 * r * fn
    return (r * sum_iou / denom) if denom > 0 else 0.0


def evaluate_prob_dict(prob: dict[str, np.ndarray], fold: int | None = None, gt_df=None,
                       keep: dict | None = None, r_sparse: float = 0.77,
                       verbose: bool = False) -> dict:
    """KEEP-rule a probability dict and score it with the official pooled PQ (+sparse PQ')."""
    df, st = prob_to_submission(prob, keep, verbose=verbose)
    res = evaluate(df, fold=fold, gt_df=gt_df)
    out = {k: res[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn", "n_pred")}
    out["pq_sparse"] = sparse_pq(out, r_sparse)
    out["n_images_pred"] = st["n_images_with_pred"]
    out["n_instances"] = st["n_instances"]
    return out


# ======================================================================================
# 4. evaluation helpers (fold0 val / OOF tuning)
# ======================================================================================
def load_fold_gt(fold: int):
    """Official pooled GT for one grouped-CV fold -> gt dataframe restricted to val stems."""
    folds = json.loads((BASE / "work" / "cache" / "folds.json").read_text(encoding="utf-8"))
    val = sorted(folds["folds"][str(fold)])
    gt_all = P.gt_df_from_coco(BASE / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" /
                               "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    return P.gt_df_for_stems(gt_all, val)


def evaluate(submission_df, fold: int | None = None, gt_df=None) -> dict:
    """Official pooled PQ of a submission dataframe on one fold (or explicit gt)."""
    if gt_df is None:
        if fold is None:
            raise ValueError("need fold or gt_df")
        gt_df = load_fold_gt(fold)
    return P.evaluate(gt_df, submission_df, backend="pycocotools", return_per_image=False)


# ======================================================================================
# 5. self-test
# ======================================================================================
def selftest() -> int:
    print("\n[prob_ensemble] SELF-TEST")
    h = w = 512
    prob = np.zeros((h, w), dtype=np.uint8)
    prob[10:30, 10:30] = 255          # A: strong large blob -> kept
    prob[40:50, 40:50] = 255          # B: 100 px < min_area -> dropped
    prob[60:80, 60:80] = 250          # C: max 250 <= thr_hi(252) -> dropped
    prob[100:140, 100:140] = 200      # D: mean < 0.85 -> dropped
    prob[100:104, 100:140] = 255
    prob[136:140, 100:140] = 255
    keep = dict(t_lo=0.35, t_hi=0.99, min_area=200, min_meanp=0.85)
    lab, _ = keep_rule_instances(prob, keep, return_labels=True)
    n_kept = int(lab.max())
    masks = keep_rule_instances(prob, keep)
    checks = {
        "only region A kept": n_kept == 1,
        "kept area == 400": int(sum(int(m.sum()) for m in masks)) == 400,
        "masks disjoint": int(np.sum(np.stack(masks), axis=0).max()) <= 1 if masks else True,
    }
    rles = instance_masks_to_rles(masks, h, w)
    checks["RLE decode roundtrip"] = all(
        np.array_equal(P.decode_rle(r, h, w).astype(bool), m) for r, m in zip(rles, masks))
    p2 = prob.copy(); p2[10:30, 10:30] = 51
    fused_mean = fuse_probs([{"a": prob}, {"a": p2}], mode="mean")["a"]
    checks["mean fusion"] = abs(int(fused_mean[15, 15]) - (255 + 51) // 2) <= 1
    fused_max = fuse_probs([{"a": prob}, {"a": p2}], mode="max")["a"]
    checks["max fusion"] = int(fused_max[15, 15]) == 255
    fused_w = fuse_probs([{"a": prob}, {"a": p2}], weights=[3.0, 1.0], mode="wmean")["a"]
    checks["weighted fusion"] = abs(int(fused_w[15, 15]) - int(round((255 * 3 + 51) / 4))) <= 1
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.npz"
        save_prob_npz({"s1": prob, "s2": p2}, p, packed=True)
        loaded = load_prob_npz(p)
        checks["npz packed roundtrip"] = (set(loaded) == {"s1", "s2"}
                                          and np.array_equal(loaded["s1"], prob))
        p2f = Path(td) / "y.npz"
        save_prob_npz({"s1": prob}, p2f, packed=False)
        checks["npz per-stem roundtrip"] = np.array_equal(load_prob_npz(p2f)["s1"], prob)
    df, st = prob_to_submission({"img1": prob}, keep, verbose=False)
    checks["submission rows == 1"] = len(df) == 1 and df.iloc[0]["filament_id"] == "img1_1"
    checks["submission RLE decodable"] = np.array_equal(
        P.decode_rle(df.iloc[0]["segmentation_rle"], h, w).astype(bool), masks[0])
    ok = True
    for name, cond in checks.items():
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ======================================================================================
# 6. CLI
# ======================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="probability-map ensemble + KEEP rule -> submission")
    ap.add_argument("--prob-npz", nargs="+", default=None, help="one or more probability dumps")
    ap.add_argument("--multi-npz", default=None,
                    help="single npz containing several scales (probs_s*/probs(N,..))")
    ap.add_argument("--multi-dir", default=None,
                    help="directory with one npz per scale (file names carry the scale)")
    ap.add_argument("--scale-weights", nargs="*", type=float, default=None,
                    help="weights aligned with scale names sorted numerically (default equal)")
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--fuse", default="mean", choices=["mean", "wmean", "median", "max"])
    ap.add_argument("--keep", nargs=4, type=float, default=None,
                    metavar=("T_LO", "T_HI", "MIN_AREA", "MIN_MEANP"),
                    help="KEEP rule thresholds (default 0.35 0.99 200 0.85)")
    ap.add_argument("--out", default=None, help="output submission csv")
    ap.add_argument("--stats-json", default=None)
    ap.add_argument("--height", type=int, default=H)
    ap.add_argument("--width", type=int, default=W)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval-fold", type=int, default=None,
                    help="evaluate the produced submission on this grouped-CV fold")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.prob_npz and not args.multi_npz and not args.multi_dir:
        ap.error("--prob-npz / --multi-npz / --multi-dir is required (or use --selftest)")
    keep = dict(DEFAULT_KEEP)
    if args.keep is not None:
        keep.update(dict(t_lo=args.keep[0], t_hi=args.keep[1],
                         min_area=int(args.keep[2]), min_meanp=args.keep[3]))
    if args.multi_npz or args.multi_dir:
        scales = load_multiscale(args.multi_npz or args.multi_dir)
        names = sorted(scales, key=lambda s: int(s.lstrip("s")) if s.lstrip("s").isdigit() else 0)
        weights = args.scale_weights
        if weights is None:
            weights = [1.0] * len(names)
        if len(weights) != len(names):
            ap.error(f"--scale-weights has {len(weights)} values but {len(names)} scales found: {names}")
        print(f"[prob_ensemble] multi-scale: {names} weights={weights} | " +
              ", ".join(f"{n}:{len(scales[n])} imgs" for n in names))
        prob_dicts = [scales[n] for n in names]
    else:
        prob_dicts = [load_prob_npz(p) for p in args.prob_npz]
        print(f"[prob_ensemble] loaded {len(prob_dicts)} dumps: " +
              ", ".join(f"{Path(p).name}({len(d)} imgs)" for p, d in zip(args.prob_npz, prob_dicts)))
        weights = args.weights
    fused = fuse_probs(prob_dicts, weights=weights, mode=args.fuse,
                       h=args.height, w=args.width)
    print(f"[prob_ensemble] fused {len(fused)} images, KEEP={keep}")
    df, st = prob_to_submission(fused, keep, h=args.height, w=args.width, limit=args.limit)
    if args.eval_fold is not None:
        res = P.evaluate(load_fold_gt(args.eval_fold), df, backend="pycocotools",
                         return_per_image=False)
        st["eval"] = {k: res[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn", "n_pred")}
        print(f"[prob_ensemble] fold{args.eval_fold} PQ={res['pq']:.5f} SQ={res['sq']:.5f} "
              f"RQ={res['rq']:.5f} TP={res['tp']} FP={res['fp']} FN={res['fn']}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[prob_ensemble] wrote {out} ({len(df)} rows, {out.stat().st_size / 1e6:.2f} MB)")
    if args.stats_json:
        Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
        if len(st["per_image_instances"]) > 200:
            st["per_image_instances"] = st["per_image_instances"][:200]
        Path(args.stats_json).write_text(json.dumps(st, indent=2), encoding="utf-8")
        print(f"[prob_ensemble] stats -> {args.stats_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
