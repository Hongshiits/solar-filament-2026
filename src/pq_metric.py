"""Official Panoptic-Quality (PQ) metric for the Solar Filament Segmentation
Challenge 2026 -- a step-by-step replica of the organiser's
notes/nb_text/self-evaluation-notebook.py.

Reference implementation (organiser self-evaluation notebook)
--------------------------------------------------------------
* GT filament_id     = "{annotator_id}-{stem}_{k}" (one row per GT filament)
* Pred filament_id   = "{stem}_{k}"
* For every *annotator-image* (one annotation pass over one physical jpeg):
      - decode its GT masks and the predicted masks of the same physical image
      - compute the pairwise IoU matrix (torch matmul on 0/1 float32 layers)
      - hit = iou > 0.5 (strict); every hit pair is a TP
      - preds with no hit  -> FP,  GTs with no hit -> FN
* Dataset aggregate:
      PQ = sum(IoU of TP pairs) / (|TP| + 0.5*|FP| + 0.5*|FN|)
      SQ = sum(IoU of TP pairs) / |TP|,  RQ = |TP| / (|TP| + .5FP + .5FN)
  Images whose GT set is empty never enter the loop (the annotation-image list
  is built from GT only), so predictions on a stem with no GT row are ignored.
  Empty denominator -> PQ = 0.0.

Numerical equivalence
---------------------
The official notebook turns the decoded layers into torch.float32 and uses
torch.matmul.  Intersections and areas are therefore *exact integers*
(<= 2048*2048 = 2**22 < 2**24), and the IoU is a correctly rounded float32
division.  Our numpy backend reproduces that bit-for-bit:
  * intersection: np.packbits + np.bitwise_count -> exact int64
  * IoU: np.float32(inter) / np.float32(union) -> same float32 values
The optional pycocotools backend uses mask_util.iou (exact integer
intersections, division in float64, cast to float32).  Because every achievable
union is < 2**25, a float32 rounding midpoint can never be hit, so the cast is
provably identical to the official float32 division (verified on real data in
src/test_pq_metric.py).

API
---
    from pq_metric import evaluate, gt_df_from_coco, encode_masks, decode_rle

    gt_df   = gt_df_from_coco(COCO_JSON)               # (filament_id, segmentation_rle)
    pred_df = pd.DataFrame({"filament_id": [...], "segmentation_rle": [...]})
    res = evaluate(gt_df, pred_df)                     # dict: pq/sq/rq/tp/fp/fn/per_image
    print(format_report(res))

Command line
------------
    python src/pq_metric.py --gt-json <coco.json> --pred-csv <submission.csv>
    python src/pq_metric.py --gt-csv <gt.csv> --pred-csv <submission.csv> --backend numpy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

H = W = 2048
IOU_THRESHOLD = 0.5

try:  # pycocotools is optional: the numpy backend gives bit-identical results
    from pycocotools import mask as _mask_utils
    HAVE_PYOCOCOTOOLS = True
except Exception:  # pragma: no cover - only when pycocotools is absent
    _mask_utils = None
    HAVE_PYOCOCOTOOLS = False

_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")

__all__ = [
    "H", "W", "IOU_THRESHOLD", "HAVE_PYOCOCOTOOLS",
    "counts_from_compressed", "compressed_from_counts", "encode_mask", "encode_masks",
    "mask_area", "decode_rle", "rles_to_layers",
    "get_overlap_matrices", "get_overlap_df", "get_pq_score", "evaluate",
    "summarize_overlap", "format_report", "gt_df_from_coco", "gt_stems", "gt_df_for_stems",
]


# ======================================================================================
# 1. compressed COCO RLE codec (standalone; mirrors pycocotools' common/maskApi.c)
# ======================================================================================
def _as_bytes(counts) -> bytes:
    if isinstance(counts, bytes):
        return counts
    if isinstance(counts, str):
        return counts.encode("ascii")
    if isinstance(counts, (bytearray, memoryview)):
        return bytes(counts)
    raise TypeError(f"unsupported RLE counts type: {type(counts)!r}")


def counts_from_compressed(counts) -> np.ndarray:
    """Decode a compressed COCO RLE counts string into absolute run lengths.

    Mirrors maskApi.c:rleFrString (identical in pycocotools 2.0.0 .. 2.0.11):
    LEB128-like 5-bit chunks, and for index i >= 3 the stored value is a delta
    against cnts[i-2] (same-parity runs).  The first three counts are stored
    verbatim.
    """
    b = _as_bytes(counts)
    n = len(b)
    vals: list[int] = []
    p = 0
    while p < n:
        x = 0
        k = 0
        more = True
        while more:
            c = b[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = (c & 0x20) != 0
            p += 1
            k += 1
            if (not more) and (c & 0x10):
                x |= -1 << (5 * k)
        if len(vals) > 2:
            x += vals[len(vals) - 2]
        vals.append(int(x))
    return np.asarray(vals, dtype=np.int64)


def compressed_from_counts(counts) -> str:
    """Encode absolute run lengths as a compressed COCO RLE counts string.

    Mirrors maskApi.c:rleToString."""
    cnts = [int(c) for c in np.asarray(counts).reshape(-1).tolist()]
    out: list[str] = []
    for i, c in enumerate(cnts):
        x = c
        if i > 2:
            x -= cnts[i - 2]
        while True:
            ch = x & 0x1F
            x >>= 5  # arithmetic shift (python ints are arbitrary precision)
            more = (x != -1) if (ch & 0x10) else (x != 0)
            if more:
                ch |= 0x20
            out.append(chr(ch + 48))
            if not more:
                break
    return "".join(out)


def _run_lengths(flat: np.ndarray) -> np.ndarray:
    """Run lengths of a 1-D binary array, COCO convention (starts with 0-runs)."""
    flat = np.asarray(flat).reshape(-1)
    n = flat.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    flat = (flat != 0).astype(np.uint8, copy=False)
    if n == 1:
        return np.asarray([0, 1] if flat[0] else [1], dtype=np.int64)
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    pos = np.empty(change.size + 2, dtype=np.int64)
    pos[0] = 0
    pos[1:-1] = change
    pos[-1] = n
    lens = np.diff(pos)
    if flat[0] == 0:
        return lens
    return np.concatenate(([0], lens))


def encode_mask(mask) -> str:
    """Encode a 2-D binary mask as a compressed COCO RLE counts string (ASCII).

    Masks are flattened in column-major (Fortran) order, as COCO/Pycocotools
    does.  The result is the exact string that goes into segmentation_rle.
    """
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {m.shape}")
    flat = m.reshape(-1, order="F")
    return compressed_from_counts(_run_lengths(flat))


def encode_masks(masks: Iterable) -> list[str]:
    """Encode an iterable of 2-D binary masks into compressed RLE strings."""
    return [encode_mask(m) for m in masks]


def mask_area(counts, height: int | None = None, width: int | None = None) -> int:
    """Number of foreground pixels of an RLE (sum of the 1-runs)."""
    c = np.asarray(counts_from_compressed(counts))
    area = int(c[1::2].sum())
    if height is not None and width is not None:
        if int(c.sum()) != int(height) * int(width):
            raise ValueError(
                f"RLE has {int(c.sum())} pixels, expected {height}*{width}={height * width}")
    return area


def decode_rle(counts, height: int = H, width: int = W) -> np.ndarray:
    """Decode one compressed COCO RLE string into a uint8 (height, width) mask."""
    c = counts_from_compressed(counts)
    total = int(height) * int(width)
    if int(c.sum()) != total:
        raise ValueError(f"RLE pixel count {int(c.sum())} != {height}*{width}={total}")
    out = np.zeros(total, dtype=np.uint8)
    pos = 0
    for i in range(c.size):
        run = int(c[i])
        if i & 1:
            out[pos:pos + run] = 1
        pos += run
    # painted in column-major pixel order -> reshape to the image in Fortran order
    return out.reshape((int(height), int(width)), order="F")


def rles_to_layers(rles: Sequence, height: int = H, width: int = W) -> np.ndarray:
    """Decode RLE strings into a float32 (n, height, width) stack.

    Same contract as the organiser's rles_to_layers (float32 layers in {0, 1},
    empty input -> (0, height, width))."""
    if not rles:
        return np.zeros((0, int(height), int(width)), dtype=np.float32)
    stack = np.empty((len(rles), int(height), int(width)), dtype=np.float32)
    for i, rle in enumerate(rles):
        stack[i] = decode_rle(rle, height, width)
    return stack


# ======================================================================================
# 2. overlap matrices (numpy replica of get_overlap_matrices)
# ======================================================================================
def _pack_bits(layers: np.ndarray) -> np.ndarray:
    """(n, H, W) binary stack -> (n, ceil(H*W/8)) packed uint8 (row-major pixels)."""
    stack = np.asarray(layers)
    n = int(stack.shape[0])
    hw = int(stack.shape[1]) * int(stack.shape[2])
    if stack.dtype != np.uint8:
        stack = (stack != 0)
    return np.packbits(stack.reshape(n, hw), axis=1)


def _px_count(layers: np.ndarray) -> np.ndarray:
    """Exact per-layer foreground pixel counts (int64)."""
    stack = np.asarray(layers)
    n = int(stack.shape[0])
    hw = int(stack.shape[1]) * int(stack.shape[2])
    if stack.dtype == np.uint8:
        return stack.reshape(n, hw).sum(axis=1, dtype=np.int64)
    return (stack.reshape(n, hw) != 0).sum(axis=1, dtype=np.int64)


def _intersection_from_packed(gt_p: np.ndarray, pred_p: np.ndarray) -> np.ndarray:
    """Exact intersection counts from two packed-bit stacks (see _pack_bits)."""
    n_gt = int(gt_p.shape[0])
    n_pred = int(pred_p.shape[0])
    inter = np.zeros((n_gt, n_pred), dtype=np.int64)
    if n_gt == 0 or n_pred == 0:
        return inter
    pad = (-gt_p.shape[1]) % 8
    if pad:
        gt_p = np.pad(gt_p, ((0, 0), (0, pad)))
        pred_p = np.pad(pred_p, ((0, 0), (0, pad)))
    if _HAS_BITWISE_COUNT:
        gt_u = gt_p.view(np.uint64)      # (n_gt, P/8)
        pred_u = pred_p.view(np.uint64)  # (n_pred, P/8)
        for i in range(n_gt):
            inter[i] = np.bitwise_count(gt_u[i] & pred_u).sum(axis=1)
        return inter
    gt_b = np.unpackbits(gt_p, axis=1).astype(bool)  # pragma: no cover
    pred_b = np.unpackbits(pred_p, axis=1).astype(bool)
    for i in range(n_gt):  # pragma: no cover - slow fallback for numpy < 2.0
        for j in range(n_pred):
            inter[i, j] = np.count_nonzero(gt_b[i] & pred_b[j])
    return inter


def _intersection_matrix(gt_layers: np.ndarray, pred_layers: np.ndarray) -> np.ndarray:
    """Exact integer intersection counts between two binary layer stacks.

    (n_gt, H, W) x (n_pred, H, W) -> int64 (n_gt, n_pred).
    Uses bit packing + popcount when available (fast), otherwise a pairwise loop.
    """
    return _intersection_from_packed(_pack_bits(gt_layers), _pack_bits(pred_layers))


def _overlap_from_packed(gt_p, pred_p, gt_areas, pred_areas):
    """IoU / Dice matrices (float32) from packed stacks + exact areas.

    This is the bit-exact numpy replica of the organiser's torch versions:
    intersection/areas are exact integers, the quotients are float32 divisions.
    """
    inter = _intersection_from_packed(gt_p, pred_p)
    gt_a = np.asarray(gt_areas, dtype=np.int64)
    pred_a = np.asarray(pred_areas, dtype=np.int64)
    inter32 = inter.astype(np.float32)
    union = gt_a.astype(np.float32)[:, None] + pred_a.astype(np.float32)[None, :] - inter32
    iou = np.zeros(inter.shape, dtype=np.float32)
    np.divide(inter32, union, out=iou, where=(union != 0))
    dice = np.zeros(inter.shape, dtype=np.float32)
    denom = gt_a.astype(np.float32)[:, None] + pred_a.astype(np.float32)[None, :]
    np.divide(2.0 * inter32, denom, out=dice, where=(denom != 0))
    return iou, dice


def _decode_stack_uint8(rles: Sequence, height: int = H, width: int = W):
    """Decode RLE strings into a uint8 (n, H, W) stack + exact int64 areas.

    Painting goes through a column-major flat buffer, so it is ~2x faster than
    materialising float32 masks.  Used by the numpy backend only.
    """
    n = len(rles)
    h = int(height)
    w = int(width)
    hw = h * w
    stack = np.zeros((n, h, w), dtype=np.uint8)
    areas = np.zeros(n, dtype=np.int64)
    for i, rle in enumerate(rles):
        cnts = counts_from_compressed(rle)
        if int(cnts.sum()) != hw:
            raise ValueError(f"RLE pixel count {int(cnts.sum())} != {h}*{w}={hw}")
        areas[i] = int(cnts[1::2].sum())
        buf = np.zeros(hw, dtype=np.uint8)
        pos = 0
        for k in range(cnts.size):
            run = int(cnts[k])
            if k & 1:
                buf[pos:pos + run] = 1
            pos += run
        stack[i] = buf.reshape((h, w), order="F")
    return stack, areas


def get_overlap_matrices(gt_layers, pred_layers):
    """Pairwise IoU / Dice matrices, float32, matching the organiser's function.

    Args:
        gt_layers:   (n_gt, H, W) binary stack.
        pred_layers: (n_pred, H, W) binary stack.

    Returns:
        (iou_matrix, dice_matrix), both float32 shaped (n_gt, n_pred);
        entries are 0 where the union is empty.
    """
    gt = np.asarray(gt_layers)
    pred = np.asarray(pred_layers)
    return _overlap_from_packed(_pack_bits(gt), _pack_bits(pred),
                                _px_count(gt), _px_count(pred))


def _rle_bbox_dicts(rles: Sequence, height: int, width: int) -> list[dict]:
    return [{"size": [int(height), int(width)], "counts": _as_bytes(r)} for r in rles]


def _iou_matrix_pycocotools(gt_rles: Sequence, pred_rles: Sequence,
                            height: int, width: int) -> np.ndarray:
    """IoU matrix via pycocotools' C implementation (fast path)."""
    if not HAVE_PYOCOCOTOOLS:  # pragma: no cover
        raise RuntimeError("pycocotools backend requested but pycocotools is not installed")
    gt = _rle_bbox_dicts(gt_rles, height, width)
    pred = _rle_bbox_dicts(pred_rles, height, width)
    iscrowd = [0] * len(pred)  # one flag per "gt" (2nd) argument in pycocotools' iou(dt, gt, iscrowd)
    return _mask_utils.iou(gt, pred, iscrowd).astype(np.float32)


# ======================================================================================
# 3. DataFrame plumbing (annotation-image / image grouping, official semantics)
# ======================================================================================
def _gt_annotator_image_ids(gt_df: pd.DataFrame) -> pd.Series:
    """'{annotator}-{stem}_{k}' -> '{annotator}-{stem}' (split on 1st '_')."""
    return gt_df["filament_id"].astype(str).str.split("_", n=1).str[0]


def _pred_image_ids(pred_df: pd.DataFrame) -> pd.Series:
    """'{stem}_{k}' -> '{stem}' (split on 1st '_')."""
    return pred_df["filament_id"].astype(str).str.split("_", n=1).str[0]


def _group_rles(ids: pd.Series, rles: pd.Series) -> dict[str, list]:
    """Ordered id -> [rle, ...] mapping preserving row order (no full decode)."""
    groups: dict[str, list] = {}
    for key, rle in zip(ids.tolist(), rles.tolist()):
        groups.setdefault(key, []).append(rle)
    return groups


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "pycocotools" if HAVE_PYOCOCOTOOLS else "numpy"
    if backend == "pycocotools" and not HAVE_PYOCOCOTOOLS:
        raise RuntimeError("backend='pycocotools' but pycocotools is not installed")
    if backend not in ("numpy", "pycocotools"):
        raise ValueError(f"unknown backend {backend!r} (expected auto/numpy/pycocotools)")
    return backend


def _iou_for_entry(gt_rles, pred_rles, height, width, backend):
    """Return (iou, dice_or_None) for one annotator-image."""
    n_gt = len(gt_rles)
    n_pred = len(pred_rles)
    if n_gt == 0 or n_pred == 0:
        iou = np.zeros((n_gt, n_pred), dtype=np.float32)
        dice = None if backend == "pycocotools" else np.zeros((n_gt, n_pred), dtype=np.float32)
        return iou, dice
    if backend == "pycocotools":
        iou = _iou_matrix_pycocotools(gt_rles, pred_rles, height, width)
        # Dice = 2I/(a+b) = 2*IoU/(1+IoU) (diagnostic only; exact up to float rounding)
        iou64 = iou.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            dice = (2.0 * iou64 / (1.0 + iou64)).astype(np.float32)
        return iou, dice
    gt_u8, gt_areas = _decode_stack_uint8(gt_rles, height, width)
    pred_u8, pred_areas = _decode_stack_uint8(pred_rles, height, width)
    return _overlap_from_packed(_pack_bits(gt_u8), _pack_bits(pred_u8), gt_areas, pred_areas)


def process_entry(annotator_image: str, gt_annotator_image_ids: pd.Series,
                  pred_image_ids: pd.Series, gt_df: pd.DataFrame, pred_df: pd.DataFrame,
                  height: int = H, width: int = W, backend: str = "auto"):
    """Same contract as the organiser's process_entry (numpy backend by default).

    Decodes the GT masks of one annotator-image and the prediction masks of the
    physical image, returning (iou_matrix, dice_matrix, n_gt, n_pred).
    """
    image_id = annotator_image.split("-", maxsplit=1)[1]
    gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_image, "segmentation_rle"].tolist()
    pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
    iou, dice = _iou_for_entry(gt_rles, pred_rles, height, width, _resolve_backend(backend))
    return iou, dice, len(gt_rles), len(pred_rles)


def get_overlap_df(gt_df: pd.DataFrame, pred_df: pd.DataFrame,
                   height: int = H, width: int = W, backend: str = "auto",
                   want_dice: bool = True, verbose: bool = False) -> pd.DataFrame:
    """Per-annotator-image overlap table (same columns as the official notebook).

    Columns: annotator_image, iou_matrix, dice_matrix, n_gt, n_pred.
    Only GT annotation-images are enumerated (predictions on stems without any GT
    row are ignored, exactly like the organiser notebook).  Masks are decoded one
    annotator-image at a time, so memory stays bounded.
    """
    for df, name in ((gt_df, "gt_df"), (pred_df, "pred_df")):
        missing = {"filament_id", "segmentation_rle"} - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns {sorted(missing)}")
    backend = _resolve_backend(backend)

    gt_ids = _gt_annotator_image_ids(gt_df)
    pred_ids = _pred_image_ids(pred_df)
    gt_groups = _group_rles(gt_ids, gt_df["segmentation_rle"])
    pred_groups = _group_rles(pred_ids, pred_df["segmentation_rle"])

    rows = []
    t0 = time.time()
    for n_done, (annotator_image, gt_rles) in enumerate(gt_groups.items(), 1):
        image_id = annotator_image.split("-", maxsplit=1)[1]
        pred_rles = pred_groups.get(image_id, [])
        iou, dice = _iou_for_entry(gt_rles, pred_rles, height, width, backend)
        if not want_dice:
            dice = None
        rows.append({
            "annotator_image": annotator_image,
            "image_id": image_id,
            "iou_matrix": iou,
            "dice_matrix": dice,
            "n_gt": len(gt_rles),
            "n_pred": len(pred_rles),
        })
        if verbose and n_done % 100 == 0:
            print(f"  [overlap] {n_done}/{len(gt_groups)} annotator-images "
                  f"({time.time() - t0:.1f}s)", file=sys.stderr)
    out = pd.DataFrame(rows, columns=["annotator_image", "image_id", "iou_matrix",
                                      "dice_matrix", "n_gt", "n_pred"])
    if verbose:
        print(f"  [overlap] done: {len(out)} annotator-images in {time.time() - t0:.1f}s",
              file=sys.stderr)
    return out


# ======================================================================================
# 4. PQ aggregation (replica of get_pq_score) + richer summary
# ======================================================================================
def fp_count_hit(hit_matrix) -> int:
    """Predictions with no qualifying GT match (empty column)."""
    return int(np.sum(np.asarray(hit_matrix).sum(axis=0) == 0))


def fn_count_hit(hit_matrix) -> int:
    """GT filaments with no qualifying prediction (empty row)."""
    return int(np.sum(np.asarray(hit_matrix).sum(axis=1) == 0))


def get_pq_score(overlap_df: pd.DataFrame, iou_threshold: float = IOU_THRESHOLD) -> float:
    """Exact replica of the organiser's get_pq_score (threshold 0.5).

    PQ = sum(IoU of TP pairs) / (|TP| + 0.5*|FP| + 0.5*|FN|), else 0.0.
    """
    tp_iou_scores: list[float] = []
    fp_count = 0
    fn_count = 0
    for row in overlap_df.itertuples(index=False):
        iou_matrix = row.iou_matrix
        n_gt = int(row.n_gt)
        n_pred = int(row.n_pred)
        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue
        hit_matrix = iou_matrix > iou_threshold
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += fp_count_hit(hit_matrix)
        fn_count += fn_count_hit(hit_matrix)
    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count
    return float(sum(tp_iou_scores) / denominator) if denominator > 0 else 0.0


def summarize_overlap(overlap_df: pd.DataFrame, iou_threshold: float = IOU_THRESHOLD,
                      return_per_image: bool = True, return_tp_ious: bool = False) -> dict:
    """Full diagnostic summary: PQ/SQ/RQ, TP/FP/FN, per-image detail."""
    tp_ious: list[float] = []
    fp = fn = 0
    n_gt_total = n_pred_total = 0
    per_image = []
    for row in overlap_df.itertuples(index=False):
        iou_matrix = np.asarray(row.iou_matrix)
        n_gt, n_pred = int(row.n_gt), int(row.n_pred)
        n_gt_total += n_gt
        n_pred_total += n_pred
        tp = fpi = fni = 0
        s = 0.0
        if n_gt == 0:
            fpi = n_pred
        elif n_pred == 0:
            fni = n_gt
        else:
            hit = iou_matrix > iou_threshold
            vals = iou_matrix[hit]
            tp = int(vals.size)
            s = float(vals.sum())
            tp_ious.extend(vals.tolist())
            fpi = fp_count_hit(hit)
            fni = fn_count_hit(hit)
        fp += fpi
        fn += fni
        if return_per_image:
            denom = tp + 0.5 * fpi + 0.5 * fni
            per_image.append({
                "annotator_image": row.annotator_image,
                "image_id": row.image_id,
                "n_gt": n_gt,
                "n_pred": n_pred,
                "tp": tp,
                "fp": fpi,
                "fn": fni,
                "sum_tp_iou": s,
                "sq_image": (s / tp) if tp else 0.0,
                "rq_image": (tp / denom) if denom > 0 else 0.0,
                "pq_image": (s / denom) if denom > 0 else 0.0,
            })
    tp_count = len(tp_ious)
    denominator = tp_count + 0.5 * fp + 0.5 * fn
    sum_iou = float(sum(tp_ious))
    pq = (sum_iou / denominator) if denominator > 0 else 0.0
    sq = (sum_iou / tp_count) if tp_count else 0.0
    rq = (tp_count / denominator) if denominator > 0 else 0.0
    res = {
        "pq": pq, "sq": sq, "rq": rq,
        "tp": tp_count, "fp": int(fp), "fn": int(fn),
        "sum_tp_iou": sum_iou, "denominator": denominator,
        "n_annotator_images": int(len(overlap_df)),
        "n_gt": int(n_gt_total), "n_pred": int(n_pred_total),
        "iou_threshold": float(iou_threshold),
    }
    if return_per_image:
        res["per_image"] = pd.DataFrame(per_image)
    if return_tp_ious:
        res["tp_ious"] = np.asarray(tp_ious, dtype=np.float64)
    return res


def evaluate(gt_df: pd.DataFrame, pred_df: pd.DataFrame,
             height: int = H, width: int = W, backend: str = "auto",
             iou_threshold: float = IOU_THRESHOLD, want_dice: bool = False,
             return_per_image: bool = True, return_tp_ious: bool = False,
             verbose: bool = False) -> dict:
    """Official PQ plus diagnostics for one gt_df / pred_df pair.

    Args:
        gt_df / pred_df: DataFrames with columns filament_id, segmentation_rle.
        backend: auto (pycocotools if available, else numpy), pycocotools
            (fast, C implementation) or numpy (bit-exact replica of torch).
        iou_threshold: official value is 0.5; other values are diagnostics only.
    Returns:
        dict with keys pq, sq, rq, tp, fp, fn, sum_tp_iou, denominator,
        n_annotator_images, n_gt, n_pred, iou_threshold, backend and optionally
        per_image (DataFrame) / tp_ious (ndarray).
    """
    overlap_df = get_overlap_df(gt_df, pred_df, height=height, width=width,
                                backend=backend, want_dice=want_dice, verbose=verbose)
    res = summarize_overlap(overlap_df, iou_threshold=iou_threshold,
                            return_per_image=return_per_image,
                            return_tp_ious=return_tp_ious)
    res["backend"] = _resolve_backend(backend)
    res["height"], res["width"] = int(height), int(width)
    return res


def format_report(res: dict, top: int = 10) -> str:
    """Human readable one-shot report string."""
    lines = [
        f"PQ = {res['pq']:.6f}   (SQ = {res['sq']:.6f}, RQ = {res['rq']:.6f})",
        f"TP = {res['tp']}   FP = {res['fp']}   FN = {res['fn']}"
        f"   denominator = {res['denominator']:.1f}",
        f"annotator-images = {res['n_annotator_images']}   GT = {res['n_gt']}   pred = {res['n_pred']}"
        f"   threshold = {res['iou_threshold']}   backend = {res.get('backend', '?')}",
    ]
    per = res.get("per_image")
    if per is not None and len(per):
        agg = per.groupby("image_id").agg(
            n_gt=("n_gt", "sum"), n_pred=("n_pred", "sum"),
            tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
            pq_image=("pq_image", "mean"),
        ).sort_values("pq_image").head(top)
        lines.append("worst physical images (mean annotator PQ):")
        for img, r in agg.iterrows():
            lines.append(f"  {img}  pq~{r['pq_image']:.3f}  gt={int(r['n_gt'])} "
                         f"pred={int(r['n_pred'])} tp={int(r['tp'])} fp={int(r['fp'])} fn={int(r['fn'])}")
    return "\n".join(lines)


# ======================================================================================
# 5. GT dataframe construction from the COCO annotation JSON
# ======================================================================================
def _polygon_rle(coco_seg, height: int, width: int) -> bytes:
    """Rasterise a COCO polygon annotation into compressed RLE counts (bytes)."""
    if not HAVE_PYOCOCOTOOLS:  # pragma: no cover
        raise RuntimeError("gt_df_from_coco needs pycocotools (exact polygon rasterisation)")
    if isinstance(coco_seg, dict):  # already an RLE dict
        return _mask_utils.merge([coco_seg])["counts"]
    polys = coco_seg
    if isinstance(polys, list) and polys and isinstance(polys[0], (int, float)):
        polys = [polys]
    rles = _mask_utils.frPyObjects(polys, int(height), int(width))
    if len(rles) == 1:
        return rles[0]["counts"]
    return _mask_utils.merge(rles)["counts"]


def gt_df_from_coco(coco_json: str | Path, height: int | None = None,
                    width: int | None = None, stems: Iterable[str] | None = None,
                    verbose: bool = False) -> pd.DataFrame:
    """Build the ground-truth DataFrame required by the official metric.

    GT filament_id = "{image_id}_{k}" where image_id is the COCO image id
    ("{annotator_id}-{stem}") and k numbers the annotations of that annotation
    pass in JSON order.  Masks are produced with pycocotools.frPyObjects -- the
    same rasterisation the organisers used (proof: every ann["area"] equals the
    rasterised pixel count, see notes/HARNESS.md).

    Args:
        coco_json: path to *_Annotations_kaggle2026_train.json.
        stems: optional physical jpeg stems to keep (filters whole annotator-images).
    """
    with open(coco_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    images = data["images"]
    if height is None:
        height = int(images[0]["height"]) if images else H
    if width is None:
        width = int(images[0]["width"]) if images else W
    keep = None
    if stems is not None:
        keep = set(stems)
    rows = []
    counter: dict[str, int] = {}
    t0 = time.time()
    for ann in data["annotations"]:
        image_id = str(ann["image_id"])
        if keep is not None:
            stem = image_id.split("-", maxsplit=1)[-1]
            if stem not in keep:
                continue
        k = counter.get(image_id, 0) + 1
        counter[image_id] = k
        counts = _polygon_rle(ann["segmentation"], height, width)
        rows.append({
            "filament_id": f"{image_id}_{k}",
            "segmentation_rle": counts.decode("ascii") if isinstance(counts, bytes) else str(counts),
            "area": float(ann.get("area", -1)),
            "category_id": int(ann.get("category_id", -1)),
            "annotation_id": str(ann.get("id", "")),
        })
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle",
                                     "area", "category_id", "annotation_id"])
    if verbose:
        print(f"[gt_df_from_coco] {len(df)} GT filaments / "
              f"{df['filament_id'].str.split('_', n=1).str[0].nunique()} annotator-images "
              f"in {time.time() - t0:.1f}s", file=sys.stderr)
    return df


def gt_stems(gt_df: pd.DataFrame) -> list[str]:
    """Sorted physical jpeg stems covered by a GT dataframe."""
    annot = _gt_annotator_image_ids(gt_df)
    return sorted({a.split("-", maxsplit=1)[-1] for a in annot.unique()})


def gt_df_for_stems(gt_df: pd.DataFrame, stems: Iterable[str]) -> pd.DataFrame:
    """Restrict a GT dataframe to a set of physical jpeg stems (OOF evaluation)."""
    keep = set(stems)
    stems_of_rows = _gt_annotator_image_ids(gt_df).str.split("-", n=1).str[-1]
    return gt_df.loc[stems_of_rows.isin(keep)].reset_index(drop=True)


# ======================================================================================
# 6. CLI
# ======================================================================================
def _read_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".json":
        return gt_df_from_coco(p, verbose=True)
    return pd.read_csv(p, dtype={"filament_id": str, "segmentation_rle": str})


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Official PQ metric for filament segmentation 2026")
    ap.add_argument("--gt-json", help="COCO annotation JSON (train GT)")
    ap.add_argument("--gt-csv", help="prepared GT csv with filament_id,segmentation_rle")
    ap.add_argument("--pred-csv", required=True, help="prediction csv / submission.csv")
    ap.add_argument("--backend", default="auto", choices=["auto", "numpy", "pycocotools"])
    ap.add_argument("--threshold", type=float, default=IOU_THRESHOLD,
                    help="IoU threshold (official = 0.5; other values are diagnostics)")
    ap.add_argument("--stems", default=None,
                    help="comma separated physical stems to evaluate (default: all GT)")
    ap.add_argument("--dump-per-image", default=None, help="optional csv output for per-image detail")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if not args.gt_json and not args.gt_csv:
        ap.error("one of --gt-json / --gt-csv is required")
    stems = [s.strip() for s in args.stems.split(",") if s.strip()] if args.stems else None
    if args.gt_json:
        gt_df = gt_df_from_coco(args.gt_json, stems=stems, verbose=args.verbose)
    else:
        gt_df = pd.read_csv(args.gt_csv, dtype={"filament_id": str, "segmentation_rle": str})
        if stems is not None:
            gt_df = gt_df_for_stems(gt_df, stems)
    pred_df = pd.read_csv(args.pred_csv, dtype={"filament_id": str, "segmentation_rle": str})
    pred_df = pred_df.dropna(subset=["filament_id", "segmentation_rle"])

    res = evaluate(gt_df, pred_df, backend=args.backend, iou_threshold=args.threshold,
                   return_per_image=True, verbose=args.verbose)
    print(format_report(res))
    if args.dump_per_image:
        Path(args.dump_per_image).parent.mkdir(parents=True, exist_ok=True)
        res["per_image"].to_csv(args.dump_per_image, index=False)
        print(f"[pq_metric] per-image detail -> {args.dump_per_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
