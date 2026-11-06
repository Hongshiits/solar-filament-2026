"""Evaluate a prediction CSV with the organisers' pooled Panoptic Quality semantics.

This is the evaluation entry point used by the demo notebooks: it reproduces the scores in the report
and the organisers' self-evaluation notebook (TP = every GT/prediction pair with IoU > 0.5, pooled over
annotator records) and additionally prints the Dice/IoU distributions and the one-to-many / many-to-one
degree counts required by the evaluation rubric.

Usage
-----
    python src/eval_demo.py --pred assets/dariush_fold0_oof.csv \
        --gt-json /kaggle/input/.../MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --stems-json assets/fold0_images.json

The default --stems-json restricts evaluation to the grouped fold-0 images, so the predictions of the
shipped out-of-fold file are honest out-of-fold predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

H = W = 2048


def rles_to_layers(rles: list[str]) -> np.ndarray:
    if not rles:
        return np.zeros((0, H, W), dtype=np.float32)
    masks = mask_util.decode([{"size": [H, W], "counts": r.encode("ascii") if isinstance(r, str) else r} for r in rles])
    return masks.transpose(2, 0, 1).astype(np.float32)


def overlap_matrices(gt_layers: np.ndarray, pred_layers: np.ndarray):
    n_gt, n_pred = gt_layers.shape[0], pred_layers.shape[0]
    gt_flat, pred_flat = gt_layers.reshape(n_gt, -1), pred_layers.reshape(n_pred, -1)
    inter = gt_flat @ pred_flat.T
    gt_a, pred_a = gt_flat.sum(axis=1)[:, None], pred_flat.sum(axis=1)[None, :]
    union = gt_a + pred_a - inter
    safe_union = np.where(union == 0, 1.0, union)
    iou = np.where(union == 0, 0.0, inter / safe_union)
    denom = np.where(gt_a + pred_a == 0, 1.0, gt_a + pred_a)
    dice = np.where(union == 0, 0.0, 2 * inter / denom)
    return iou, dice


def _encode_annotation(seg) -> str:
    """Rasterise one annotation exactly like the organisers' GT (pycocotools), return compressed RLE."""
    if isinstance(seg, dict):  # already RLE
        rle = {"size": seg["size"], "counts": seg["counts"]}
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
        return mask_util.encode(rle)["counts"].decode("ascii")
    # polygon list in COCO [x1, y1, x2, y2, ...] form, possibly several polygons
    rles = mask_util.frPyObjects(seg if isinstance(seg[0], list) else [seg], H, W)
    merged = mask_util.merge(rles)
    if isinstance(merged, dict):
        counts = merged["counts"]
        return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)
    return mask_util.encode(merged)["counts"].decode("ascii")


def gt_records(gt_json: dict, stems: set[str]) -> dict[str, list[list[str]]]:
    """Map physical stem -> list of annotation-record RLE lists (one list per annotator record)."""
    stem_by_image_id = {im["id"]: Path(str(im["file_name"])).stem.split("-")[-1] for im in gt_json["images"]}
    out: dict[str, dict[str, list[str]]] = {}
    for ann in gt_json["annotations"]:
        stem = stem_by_image_id.get(ann["image_id"])
        if stem is None or stem not in stems or "segmentation" not in ann:
            continue
        rle = _encode_annotation(ann["segmentation"])
        # group annotations by annotator record: the released filenames/attributes name the pass
        rec = str(ann.get("annotator_id", ann.get("attrs", {}).get("annotator_id", ann["image_id"])))
        out.setdefault(stem, {}).setdefault(rec, []).append(rle)
    return {s: [v for _, v in sorted(recs.items(), key=lambda kv: kv[0])] for s, recs in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--stems-json", default=None, help="optional JSON list of stems to keep (e.g. assets/fold0_images.json)")
    args = ap.parse_args()

    gt_json = json.load(open(args.gt_json, encoding="utf-8"))
    stems = set(json.load(open(args.stems_json, encoding="utf-8"))) if args.stems_json else None

    pred_df = pd.read_csv(args.pred, dtype=str)
    pred_by_stem: dict[str, list[str]] = {}
    for fid, rle in zip(pred_df["filament_id"], pred_df["segmentation_rle"]):
        stem = str(fid).rsplit("_", 1)[0]
        if stems is None or stem in stems:
            pred_by_stem.setdefault(stem, []).append(rle)

    if stems is None:
        stems = set(pred_by_stem) | {Path(str(im["file_name"])).stem.split("-")[-1] for im in gt_json["images"]}
    records = gt_records(gt_json, set(stems))

    tp_iou, fp, fn = [], 0, 0
    gt_degree, pred_degree = [], []
    iou_all, dice_all = [], []
    for stem in sorted(stems):
        for gt_rles in records.get(stem, []):
            pred_rles = pred_by_stem.get(stem, [])
            gt_l = rles_to_layers(gt_rles)
            pr_l = rles_to_layers(pred_rles)
            iou, dice = overlap_matrices(gt_l, pr_l)
            iou_all.append(iou.ravel())
            dice_all.append(dice.ravel())
            if iou.size == 0:
                fn += len(gt_rles)
                continue
            hit = iou > 0.5
            tp_iou.extend(iou[hit].tolist())
            fp += int((hit.sum(axis=0) == 0).sum())
            fn += int((hit.sum(axis=1) == 0).sum())
            hit_any = iou > 0
            gt_degree.extend(hit_any.sum(axis=1).tolist())
            pred_degree.extend(hit_any.sum(axis=0).tolist())

    tp = len(tp_iou)
    den = tp + 0.5 * fp + 0.5 * fn
    pq = float(sum(tp_iou) / den) if den else 0.0
    sq = float(np.mean(tp_iou)) if tp_iou else 0.0
    rq = float(tp / den) if den else 0.0
    print(f"PQ={pq:.4f}  SQ={sq:.4f}  RQ={rq:.4f}   TP={tp} FP={fp} FN={fn}")

    all_iou = np.concatenate(iou_all) if iou_all else np.zeros(0)
    all_dice = np.concatenate(dice_all) if dice_all else np.zeros(0)
    pos_iou = all_iou[all_iou > 0]
    pos_dice = all_dice[all_dice > 0]
    for name, arr in (("IoU", pos_iou), ("Dice", pos_dice)):
        if arr.size:
            print(f"{name} (non-zero pairs, n={arr.size}): mean={arr.mean():.3f} "
                  f"p05={np.percentile(arr,5):.3f} median={np.median(arr):.3f} p95={np.percentile(arr,95):.3f}")
    def degree_summary(deg, label):
        if not deg:
            return
        arr = np.asarray(deg)
        print(f"{label}: no-overlap={int((arr==0).sum())}, 1={int((arr==1).sum())}, "
              f"2={int((arr==2).sum())}, >=3={int((arr>=3).sum())}")
    degree_summary(gt_degree, "annotations matched by n predictions (one-to-many)")
    degree_summary(pred_degree, "predictions matched by n annotations (many-to-one)")
    return 0


if __name__ == "__main__":
    sys.exit(main())