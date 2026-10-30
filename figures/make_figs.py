#!/usr/bin/env python3
"""Generate the three figures of the competition report (Fig. 1-3).

Outputs (vector PDF + 300 dpi PNG, double-column width 7.0 in):
    fig1_pipeline.pdf / .png              pipeline block diagram
    fig2_data.pdf / .png                  3 full-disk observations, 2 multi-annotator zooms,
                                          and annotation statistics
    fig3_predictions.pdf / .png           2 successful and 2 failure cases of the submitted
                                          public-0.70 reproduction (public LB PQ 0.34)

Data required (official competition download, not redistributed):
    data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
    data/MAGFiLO_1.0_Kaggle_2026/train/train_images/*.jpeg
    data/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg
    submissions/pub070_kaggle_v1.csv          (prediction file of the reproduced baseline)

Everything runs on CPU: numpy / opencv / matplotlib / pycocotools only, no model inference.

Usage:
    python repo/figures/make_figs.py                 # auto-detect project layout
    python repo/figures/make_figs.py --data-root /path/to/MAGFiLO_1.0_Kaggle_2026 \
                                     --submission /path/to/submission.csv --out-dir .
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pycocotools import mask as mask_utils

# ----------------------------------------------------------------------------- config
FIG_W = 7.0                       # double-column width of the report (inches)
DPI = 300
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "axes.titlesize": 7.6,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "pdf.fonttype": 42,           # embed TrueType so the PDF is searchable/editable
    "ps.fonttype": 42,
    "savefig.dpi": DPI,
    "figure.dpi": 120,
})

# Okabe-Ito colour-blind-safe palette
C_GREEN = "#009E73"
C_MAGENTA = "#CC79A7"
C_YELLOW = "#F0E442"
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GRAY = "#666666"
C_LIGHT = "#E8EEF4"
C_LIGHT2 = "#F4EEE2"
C_LIGHT3 = "#EDEDED"
ANN_COLORS = [C_GREEN, C_MAGENTA, C_YELLOW]

IMAGE_SHAPE = (2048, 2048)


# ----------------------------------------------------------------------------- paths
def resolve_paths(args):
    here = Path(__file__).resolve()
    repo_root = here.parents[1]                      # .../repo
    project_root = repo_root.parent                  # .../filament-segmentation-2026 (local layout)
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is None:
        for cand in (repo_root / "data" / "MAGFiLO_1.0_Kaggle_2026",
                     project_root / "data" / "MAGFiLO_1.0_Kaggle_2026"):
            if cand.is_dir():
                data_root = cand
                break
    if data_root is None or not data_root.is_dir():
        raise FileNotFoundError("competition data not found; pass --data-root")
    submission = Path(args.submission) if args.submission else None
    if submission is None:
        for cand in (repo_root / "submissions" / "pub070_kaggle_v1.csv",
                     project_root / "submissions" / "pub070_kaggle_v1.csv"):
            if cand.is_file():
                submission = cand
                break
    if submission is None or not submission.is_file():
        raise FileNotFoundError("prediction CSV not found; pass --submission")
    out_dir = Path(args.out_dir) if args.out_dir else here.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return repo_root, project_root, data_root, submission, out_dir


# ----------------------------------------------------------------------------- helpers
def stretch(img):
    """Robust display stretch: 1st-99.6th percentile of the solar disk only."""
    disk = img > 8
    lo = float(np.percentile(img[disk], 1.0))
    hi = float(np.percentile(img[disk], 99.6))
    return (np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1) * 255).astype(np.uint8)


def ann_rle(ann, height=2048, width=2048):
    """COCO polygon annotation -> compressed RLE (same rasterisation as src/pq_metric.py)."""
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        return seg
    rles = mask_utils.frPyObjects(seg, height, width)
    return mask_utils.merge(rles) if isinstance(rles, list) else rles


def rle_to_mask(counts, height=2048, width=2048):
    return mask_utils.decode({"size": [height, width], "counts": counts.encode("utf-8")})


def mask_contours(mask):
    """External contours of a binary mask, in (x, y) image coordinates."""
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c[:, 0, :] for c in contours if len(c) >= 8]


def draw_contours(ax, mask, color, lw=1.0, offset=(0, 0), halo=True):
    """Draw mask contours with a thin dark halo for readability on bright/dark backgrounds."""
    ox, oy = offset
    for cnt in mask_contours(mask):
        xy = cnt.astype(np.float32) + np.array([ox, oy], dtype=np.float32)
        if halo:
            ax.plot(xy[:, 0], xy[:, 1], color="black", lw=lw + 1.0, alpha=0.55, solid_capstyle="round", zorder=4)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, solid_capstyle="round", zorder=5)


def crop_box(cx, cy, half, size=2048):
    x0 = int(round(max(0, min(size - 2 * half, cx - half))))
    y0 = int(round(max(0, min(size - 2 * half, cy - half))))
    return x0, y0, 2 * half, 2 * half


def panel_label(ax, letter):
    """Panel letter inside the axes, top-left, with a translucent backing box."""
    ax.text(0.015, 0.975, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold", zorder=20,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def save(fig, out_dir, stem):
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, facecolor="white")           # exact figure size: 7.0 in wide
    fig.savefig(png, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  wrote {pdf.name} ({pdf.stat().st_size/1024:.0f} kB) and {png.name}")


# ----------------------------------------------------------------------------- Fig 1
def make_fig1(out_dir):
    print("[fig1] pipeline block diagram")
    fig = plt.figure(figsize=(FIG_W, 2.95))
    ax = fig.add_axes([0, 0, 1, 1])          # full-bleed axes: 1 unit = 1% of 7 in
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge="#4A4A4A", fontsize=7.0, bold_first=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
                                    linewidth=1.0, edgecolor=edge, facecolor=face, zorder=2))
        if bold_first:
            lines = text.split("\n")
            ax.text(x + w / 2, y + h - 3.0, lines[0], ha="center", va="top",
                    fontsize=fontsize + 0.5, fontweight="bold", zorder=3)
            ax.text(x + w / 2, y + h - 8.0, "\n".join(lines[1:]), ha="center", va="top",
                    fontsize=fontsize, linespacing=1.3, zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=fontsize, linespacing=1.3, zorder=3)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                     lw=1.1, color="#333333", shrinkA=1, shrinkB=1, zorder=1))

    # row 1: input -> detector -> crop -> refiner
    box(1.5, 53, 15, 41, "Input\nGONG Hα full-disk\n2048 × 2048 px\n8-bit grayscale", C_LIGHT3, bold_first=True)
    box(18, 53, 25, 41, "Stage 1 — detector\nYOLO11s-seg (1 class)\n1024 × 1024 input\n~30 epochs · conf ≥ 0.2", C_LIGHT, bold_first=True)
    box(44.5, 53, 24, 41, "Per-instance crop\n1.8 × max(bbox)\nmin 96 px\n→ 256 × 256, 3 channels\ngray | CLAHE | seed", C_LIGHT3, bold_first=True)
    box(70, 53, 28.5, 41, "Stage 2 — refiner\nseed-conditioned U-Net\nResNet18, 8 epochs\nAdamW 3e-4\n0.45 BCE + 0.55 Dice · thr 0.65", C_LIGHT, bold_first=True)
    arrow(16.8, 73.5, 17.8, 73.5)
    arrow(43.3, 73.5, 44.3, 73.5)
    arrow(68.8, 73.5, 69.8, 73.5)

    # row 2: right-to-left post-processing
    box(70, 5, 28.5, 41, "4-flip TTA + topology guard\nmean accepted only if\narea ratio ∈ [0.6, 1.7]\nDice ≥ 0.62\ncomponents ≤ single + 1", C_LIGHT2, bold_first=True)
    box(44.5, 5, 24, 41, "Pixel-disjoint assembly\nsort by score\nclaim unclaimed pixels\nmin area 96 px", C_LIGHT3, bold_first=True)
    box(18, 5, 25, 41, "5-fold ensemble (optional)\nmask IoU ≥ 0.5 clusters\n≥ 2 folds vote, or\nhigh-confidence singleton", C_LIGHT3, bold_first=True)
    box(1.5, 5, 15, 41, "Output\nsubmission.csv\nCOCO RLE counts\n2048 × 2048\n<stem>_<k>", C_LIGHT2, bold_first=True)
    arrow(84.25, 52.5, 84.25, 47.5)
    arrow(69.8, 25.5, 68.8, 25.5)
    arrow(44.3, 25.5, 43.3, 25.5)
    arrow(17.8, 25.5, 16.8, 25.5)
    save(fig, out_dir, "fig1_pipeline")


# ----------------------------------------------------------------------------- Fig 2
def load_coco(data_root):
    coco_path = data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = json.load(open(coco_path, encoding="utf-8"))
    images = {im["id"]: im for im in coco["images"]}
    anns = defaultdict(list)
    for ann in coco["annotations"]:
        anns[ann["image_id"]].append(ann)
    by_file = defaultdict(list)
    for image_id, im in images.items():
        by_file[im["file_name"]].append(image_id)
    for ids in by_file.values():
        ids.sort()
    return images, anns, by_file


def choose_fig2_assets(images, anns, by_file):
    """Deterministic selection of the full-disk panels and the two multi-annotator zooms.

    Zoom examples: observations annotated by 3 experts, where one annotated filament can be
    matched (IoU 0.45-0.85) in all three records, i.e. the experts agree on the object but
    disagree on its extent.
    """
    multi = {fn: ids for fn, ids in by_file.items() if len(ids) >= 3}

    def bbox_overlap(a, b):
        return not (a[0] + a[2] < b[0] or b[0] + b[2] < a[0] or
                    a[1] + a[3] < b[1] or b[1] + b[3] < a[1])

    def match_in_record(ref_ann, record_anns):
        """Best-matching annotation index in one record (-1 when nothing overlaps)."""
        candidates = [(j, b) for j, b in enumerate(record_anns) if bbox_overlap(ref_ann["bbox"], b["bbox"])]
        if not candidates:
            return -1, 0.0
        iou = mask_utils.iou([ann_rle(ref_ann)], [ann_rle(b) for _, b in candidates],
                             [0] * len(candidates))[0]
        best = int(np.argmax(iou))
        return candidates[best][0], float(iou[best])

    zooms = []
    for want in ("20110306082634Lh.jpeg", "20140205180154Bh.jpeg"):
        ids = multi[want]
        chosen = None
        for k, ref in enumerate(anns[ids[0]]):
            area, bbox = float(ref["area"]), ref["bbox"]
            side = max(bbox[2], bbox[3])
            if not (600 <= area <= 2000 and 30 <= side <= 90):
                continue
            matches, ious = [k], []
            ok = True
            for ridx in range(1, len(ids)):
                j, iou = match_in_record(ref, anns[ids[ridx]])
                if j < 0 or not (0.30 <= iou <= 0.90):
                    ok = False
                    break
                matches.append(j)
                ious.append(iou)
            if ok and len(ious) == 2 and all(0.45 <= v <= 0.85 for v in ious):
                chosen = dict(stem=want, ids=ids, k=k, matches=matches, ious=ious,
                              cx=float(bbox[0] + bbox[2] / 2), cy=float(bbox[1] + bbox[3] / 2))
                break
        if chosen is None:
            raise RuntimeError(f"no usable multi-annotator zoom example found for {want}")
        zooms.append(chosen)

    full_disks = [zooms[0]["stem"], zooms[1]["stem"], "20170103030554Uh.jpeg"]
    return zooms, full_disks


def make_fig2(out_dir, data_root):
    print("[fig2] data, annotator disagreement and statistics")
    images, anns, by_file = load_coco(data_root)
    zooms, full_disks = choose_fig2_assets(images, anns, by_file)
    train_dir = data_root / "train" / "train_images"

    fig = plt.figure(figsize=(FIG_W, 4.05))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.30,
                          left=0.055, right=0.982, top=0.94, bottom=0.14)

    # ---- top row: three full-disk observations (dashed boxes mark the zoom regions)
    for col, (fn, zoom) in enumerate(zip(full_disks, zooms + [None])):
        ax = fig.add_subplot(gs[0, col])
        img = cv2.imread(str(train_dir / fn), cv2.IMREAD_GRAYSCALE)
        ax.imshow(stretch(img), cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#888888"); spine.set_linewidth(0.6)
        site = fn.split(".")[0][-2:]
        ax.set_title(f"{fn[:4]} · {site}", fontsize=7.4, pad=1.5)
        panel_label(ax, "abc"[col])
        if zoom is not None:
            half = 150
            x0, y0, w, h = crop_box(zoom["cx"], zoom["cy"], half)
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=C_MAGENTA, lw=0.9, ls=(0, (2, 1.5))))
            ax.text(x0 + 8, y0 + 18, "zoom", fontsize=6.8, color=C_MAGENTA)

    # ---- statistics: annotated filaments per annotation record
    ax = fig.add_subplot(gs[0, 3])
    per_record = [len(anns[i]) for i in images]
    ax.hist(per_record, bins=np.arange(0.5, 27.5, 1), color=C_BLUE, alpha=0.8, edgecolor="white", lw=0.3)
    med = float(np.median(per_record))
    ax.axvline(med, color=C_MAGENTA, lw=1.0, ls="--")
    ax.text(med + 0.7, ax.get_ylim()[1] * 0.80, f"median {med:.0f}", color=C_MAGENTA, fontsize=7.0)
    ax.set_xlabel("annotated filaments per record")
    ax.set_ylabel("records")
    ax.set_title("filaments per record", fontsize=7.4)
    ax.text(0.97, 0.95, f"n = {len(per_record)}", transform=ax.transAxes, ha="right", va="top", fontsize=7.0)
    ax.tick_params(length=2)
    panel_label(ax, "d")

    # ---- bottom row: two multi-annotator zooms (contours = the three experts)
    for col, zoom in enumerate(zooms):
        ax = fig.add_subplot(gs[1, col])
        img = cv2.imread(str(train_dir / zoom["stem"]), cv2.IMREAD_GRAYSCALE)
        half = 150
        x0, y0, w, h = crop_box(zoom["cx"], zoom["cy"], half)
        ax.imshow(stretch(img[y0:y0 + h, x0:x0 + w]), cmap="gray", vmin=0, vmax=255, aspect="equal")
        for ri, (iid, j) in enumerate(zip(zoom["ids"][:3], zoom["matches"][:3])):
            m = mask_utils.decode(ann_rle(anns[iid][j])).astype(bool)[y0:y0 + h, x0:x0 + w]
            draw_contours(ax, m, ANN_COLORS[ri], lw=0.9)
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#888888"); spine.set_linewidth(0.6)
        iou_txt = ", ".join(f"{v:.2f}" for v in zoom["ious"])
        ax.set_title(f"pairwise IoU {iou_txt}", fontsize=7.2, pad=1.5)
        panel_label(ax, "ef"[col])
        handles = [Line2D([0], [0], color=ANN_COLORS[i], lw=1.4, label=f"A{i + 1}") for i in range(3)]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3,
                  fontsize=7.0, frameon=False, handlelength=0.9, columnspacing=1.0, borderaxespad=0.0)

    # ---- statistics: ground-truth instance area
    ax = fig.add_subplot(gs[1, 2])
    all_anns = sum(anns.values(), [])
    areas = np.array([a["area"] for a in all_anns], dtype=float)
    short_sides = np.array([min(a["bbox"][2], a["bbox"][3]) for a in all_anns], dtype=float)
    ax.hist(np.log10(areas), bins=np.linspace(0.8, 4.7, 40), color=C_ORANGE, alpha=0.85, edgecolor="white", lw=0.3)
    ax.axvline(np.log10(np.median(areas)), color=C_MAGENTA, lw=1.0, ls="--")
    ax.set_xlabel("log$_{10}$(instance area / px$^2$)")
    ax.set_ylabel("GT instances")
    ax.set_title("instance area", fontsize=7.4)
    ax.text(0.97, 0.95, f"n = {len(areas)}", transform=ax.transAxes, ha="right", va="top", fontsize=7.0)
    ax.text(0.03, 0.88,
            f"median {np.median(areas):,.0f} px$^2$\nshort side {np.median(short_sides):.0f} px",
            transform=ax.transAxes, fontsize=6.9, va="top")
    ax.tick_params(length=2)
    panel_label(ax, "g")

    # ---- statistics: annotators per physical observation
    ax = fig.add_subplot(gs[1, 3])
    rec_counts = Counter(len(ids) for ids in by_file.values())
    labels = ["1", "2", "3"]
    values = [rec_counts.get(k, 0) for k in (1, 2, 3)]
    bars = ax.bar(labels, values, color=[C_GRAY, C_BLUE, C_GREEN], alpha=0.9, width=0.62)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, str(v), ha="center", fontsize=7.0)
    frac = 100.0 * (values[1] + values[2]) / sum(values)
    ax.set_xlabel("annotators per observation")
    ax.set_ylabel("observations")
    ax.set_title("annotators per image", fontsize=7.4)
    ax.text(0.97, 0.95, f"{frac:.0f}% multi", transform=ax.transAxes, ha="right", va="top", fontsize=7.0)
    ax.set_ylim(0, max(values) * 1.25)
    ax.tick_params(length=2)
    panel_label(ax, "h")

    fig.text(0.5, 0.015,
             "Boundary sensitivity of GT masks: 1-px erosion -> median IoU 0.71; 2-px erosion -> 0.45 (41% stay above 0.5).",
             fontsize=6.9, style="italic", color="#333333", ha="center", va="bottom")

    save(fig, out_dir, "fig2_data")


# ----------------------------------------------------------------------------- Fig 3
def load_submission(path):
    import pandas as pd
    df = pd.read_csv(path, dtype=str)
    df["stem"] = df["filament_id"].str.rsplit("_", n=1).str[0]
    masks = defaultdict(list)
    areas = defaultdict(list)
    for _, row in df.iterrows():
        m = rle_to_mask(row["segmentation_rle"]).astype(bool)
        masks[row["stem"]].append(m)
        areas[row["stem"]].append(int(m.sum()))
    return df, masks, areas


def dark_region_center(img, half=150):
    """Region of the solar disk with the most dark elongated structure (used for the miss zoom)."""
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), 25)
    dark = np.clip(blur - f, 0, None)
    yy, xx = np.mgrid[0:2048, 0:2048]
    r = np.hypot(xx - 1023.5, yy - 1023.5)
    dark = np.where((r > 150) & (r < 820), dark, 0)
    smooth = cv2.blur(dark, (2 * half, 2 * half))
    cy, cx = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    return int(cx), int(cy)


def densest_mask_center(masks):
    """Centre of the 400-px box containing the most predicted-mask centroids."""
    pts = []
    for m in masks:
        ys, xs = np.where(m)
        if len(xs):
            pts.append((xs.mean(), ys.mean()))
    pts = np.asarray(pts)
    best = (0, 1024, 1024)
    for cx, cy in pts:
        n = int(np.sum((np.abs(pts[:, 0] - cx) < 200) & (np.abs(pts[:, 1] - cy) < 200)))
        if n > best[0] or (n == best[0] and cx < best[1]):
            best = (n, cx, cy)
    return int(best[1]), int(best[2])


def make_fig3(out_dir, data_root, submission):
    print("[fig3] predictions and failure cases")
    test_dir = data_root / "test" / "test_images"
    df, masks, areas = load_submission(submission)
    all_stems = sorted(p.stem for p in test_dir.glob("*.jpeg"))
    missing = [s for s in all_stems if s not in masks]
    counts = {s: len(m) for s, m in masks.items()}
    median_n = int(np.median(list(counts.values())))

    success_a = "20190605025430Lh"          # two clearly separate, clean masks
    success_b = "20171029171010Mh"          # median number of instances
    failure_c = missing[0]                  # no prediction at all
    failure_d = max(counts, key=counts.get)  # most predictions (over-detection side)

    cases = [
        dict(stem=success_a, kind="ok", title=f"n = {counts[success_a]}", letter="a", zoom_letter="e"),
        dict(stem=success_b, kind="ok", title=f"n = {counts[success_b]} (median)", letter="b", zoom_letter="f"),
        dict(stem=failure_c, kind="miss", title="n = 0", letter="c", zoom_letter="g"),
        dict(stem=failure_d, kind="over", title=f"n = {counts[failure_d]} (max)", letter="d", zoom_letter="h"),
    ]
    for case in cases:
        img = cv2.imread(str(test_dir / f"{case['stem']}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if case["kind"] == "miss":
            cx, cy = dark_region_center(img)
        elif case["kind"] == "over":
            cx, cy = densest_mask_center(masks[case["stem"]])
        else:
            biggest = int(np.argmax(areas[case["stem"]]))
            ys, xs = np.where(masks[case["stem"]][biggest])
            cx, cy = int(xs.mean()), int(ys.mean())
        case.update(img=img, cx=cx, cy=cy)
        case["n_tiny"] = int(sum(1 for a in areas.get(case["stem"], []) if a < 400))

    fig = plt.figure(figsize=(FIG_W, 3.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], left=0.035, right=0.99, top=0.93,
                          bottom=0.135, hspace=0.24, wspace=0.14)

    def draw_full(ax, case):
        ax.imshow(stretch(case["img"]), cmap="gray", vmin=0, vmax=255, aspect="equal")
        for m in masks.get(case["stem"], []):
            draw_contours(ax, m, C_GREEN, lw=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#888888"); spine.set_linewidth(0.6)
        ax.set_title(case["title"], fontsize=7.4, pad=1.5)
        panel_label(ax, case["letter"])
        x0, y0, w, h = crop_box(case["cx"], case["cy"], 150)
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=C_MAGENTA, lw=0.9, ls=(0, (2, 1.5))))

    def draw_zoom(ax, case):
        x0, y0, w, h = crop_box(case["cx"], case["cy"], 150)
        ax.imshow(stretch(case["img"][y0:y0 + h, x0:x0 + w]), cmap="gray", vmin=0, vmax=255, aspect="equal")
        for m in masks.get(case["stem"], []):
            sub = m[y0:y0 + h, x0:x0 + w]
            if sub.any():
                draw_contours(ax, sub, C_GREEN, lw=0.85)
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#888888"); spine.set_linewidth(0.6)
        panel_label(ax, case["zoom_letter"])
        if case["kind"] == "miss":
            ax.text(0.5, 0.12, "no prediction in this region", transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=7.0, color="#B22222",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        if case["kind"] == "over":
            ax.text(0.02, 0.04, f"{case['n_tiny']} of {counts[case['stem']]} masks < 400 px$^2$",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    for col, case in enumerate(cases):
        draw_full(fig.add_subplot(gs[0, col]), case)
        draw_zoom(fig.add_subplot(gs[1, col]), case)

    fig.text(0.012, 0.73, "full disk", rotation=90, fontsize=6.9, va="center", ha="center", color="#666666")
    fig.text(0.012, 0.27, "zoom", rotation=90, fontsize=6.9, va="center", ha="center", color="#666666")
    caption = ("Predictions of the reproduced public-0.70 pipeline (public LB PQ 0.34), green contours.\n"
               "(a,b) plausible instance sets; (c) no instance returned; (d) 19 instances, 5 smaller than 400 px$^2$ (GT median 1,228).")
    fig.text(0.5, 0.008, caption, fontsize=6.9, color="#333333", ha="center", va="bottom")

    save(fig, out_dir, "fig3_predictions")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None, help="path to MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument("--submission", default=None, help="prediction CSV (submission format)")
    ap.add_argument("--out-dir", default=None, help="output directory (default: next to this script)")
    ap.add_argument("--only", default="all", choices=["all", "1", "2", "3"])
    args = ap.parse_args()

    _, _, data_root, submission, out_dir = resolve_paths(args)
    print(f"data root : {data_root}")
    print(f"submission: {submission}")
    print(f"output    : {out_dir}")
    if args.only in ("all", "1"):
        make_fig1(out_dir)
    if args.only in ("all", "2"):
        make_fig2(out_dir, data_root)
    if args.only in ("all", "3"):
        make_fig3(out_dir, data_root, submission)
    print("done.")


if __name__ == "__main__":
    main()
