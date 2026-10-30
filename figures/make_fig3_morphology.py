
"""Rebuild morphology figure in a compact 1x4 layout (GT/pred pairs side by side)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from pycocotools import mask as mask_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = Path(r"D:\PROJECT\kaggle\filament-segmentation-2026")
sys.path.insert(0, str(BASE/"src"))
import pq_metric as P
H = W = 2048
FOLDS = json.load(open(BASE/"work/cache/folds.json", encoding="utf-8"))
GT_JSON = BASE/"data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMG = BASE/"data/MAGFiLO_1.0_Kaggle_2026/train/train_images"
OUT = BASE/"docs"/"fig3_final_submission.pdf"

def load_pred(path):
    df = pd.read_csv(path, dtype=str)
    out = {}
    for fid, rle in zip(df.filament_id, df.segmentation_rle):
        stem = fid.rsplit("_", 1)[0]
        out.setdefault(stem, []).append(mask_utils.decode({"counts": rle.encode("ascii"), "size": [H, W]}).astype(bool))
    return out

def load_gt(stems):
    gt = P.gt_df_for_stems(P.gt_df_from_coco(GT_JSON), stems)
    gt["aid"] = gt["filament_id"].astype(str).str.rsplit("_", n=1).str[0]
    out = {}
    for aid, grp in gt.groupby("aid"):
        stem = aid.split("-", 1)[-1]
        masks = [mask_utils.decode({"counts": r.encode("ascii") if isinstance(r, str) else r, "size": [H, W]}).astype(bool) for r in grp["segmentation_rle"].tolist()]
        out.setdefault(stem, []).append(masks)
    return out

def overlay(ax, img, masks, color):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    canvas = np.zeros((H, W, 4), dtype=np.float32)
    for m in masks: canvas[m] = (*color, 0.45)
    ax.imshow(canvas)
    ys, xs = np.where(np.any(masks, axis=0))
    pad = 70
    ax.set_xlim(max(0, xs.min()-pad), min(W, xs.max()+pad))
    ax.set_ylim(min(H, ys.max()+pad), max(0, ys.min()-pad))
    ax.set_xticks([]); ax.set_yticks([])

def main():
    stems = sorted(FOLDS["folds"]["0"])
    pred = load_pred(BASE/"work/dariush_oof/pred_dariush_fold0.csv")
    gt = load_gt(stems)
    chosen = [s for s in stems if s in pred and s in gt and 3 <= len(gt[s][0]) <= 8][:2]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05))
    plt.subplots_adjust(left=0.005, right=0.995, top=0.80, bottom=0.02, wspace=0.03)
    for i, stem in enumerate(chosen):
        img = plt.imread(TRAIN_IMG/f"{stem}.jpeg")
        img = (np.asarray(img)*255).astype(np.uint8) if img.dtype != np.uint8 else img
        if img.ndim == 3: img = img[..., 0]
        overlay(axes[2*i], img, gt[stem][0], (0.1, 0.8, 0.2))
        overlay(axes[2*i+1], img, pred.get(stem, []), (0.9, 0.15, 0.15))
        axes[2*i].text(0.02, -0.03, f"({'ac'[i]})", transform=axes[2*i].transAxes, fontsize=8)
        axes[2*i+1].text(0.02, -0.03, f"({'bd'[i]})", transform=axes[2*i+1].transAxes, fontsize=8)
    legend = [Patch(facecolor=(0.1,0.8,0.2,0.45), label="ground truth (one annotator)"),
              Patch(facecolor=(0.9,0.15,0.15,0.45), label="final submission (out-of-fold)")]
    fig.legend(handles=legend, loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=1.5)
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    print("saved", OUT, OUT.stat().st_size)

if __name__ == "__main__":
    main()