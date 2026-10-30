"""Cache the MAGFiLO COCO annotations as per-annotator-image instance label maps.

Produces (all under work/cache/, write scope of the harness task):
    labels/<annotator_id>-<stem>.png   1154 x (2048x2048 uint16) instance label maps
                                       0 = background, 1..N = instances sorted by annotation id
    instances.json                     per annotator-image instance list
                                       (annotation id, category, area, bbox, spine)
    physical.json                      707 physical jpegs -> annotator records + medoid annotator
                                       (pairwise panoptic/semantic IoU of the union masks)

Rasterisation: pycocotools.frPyObjects (NOT cv2.fillPoly).
    All 8199 annotation records have ann["area"] exactly equal to the pycocotools
    rasterised pixel count (0 exceptions) -> the official GT masks were produced with
    pycocotools.  cv2.fillPoly(np.round(poly)) gives ~+18% area on small filaments and
    only ~0.85 IoU with the official rasterisation, so training targets must use the
    pycocotools rasterisation to match the evaluation GT.

Label map details
-----------------
* one PNG per annotator-image record (the same physical jpeg may have 1-3 records);
* instances are painted in ascending annotation-id order, so in the rare case where
  two polygons of one record overlap, the larger annotation id wins;
* instances.json["stats"]["overlap_px_total"] reports how many pixels were contested,
  so you can judge whether that matters (it is tiny in practice).
* medoid annotator per physical jpeg: among the 1-3 records, the one whose union mask
  has the highest mean pairwise IoU with the others (ties -> smallest annotator id).
  Single-record stems: that record is the medoid by definition.

Usage
-----
    python src/data_prep.py                 # full build (~3-6 min, ~<1 GB)
    python src/data_prep.py --limit 10      # quick smoke test
    python src/data_prep.py --no-png        # json only (skip label maps)
    python src/data_prep.py --verify-sample 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "MAGFiLO_1.0_Kaggle_2026"
COCO_JSON = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMAGES = DATA / "train" / "train_images"
CACHE = BASE / "work" / "cache"
LABELS_DIR = CACHE / "labels"
INSTANCES_JSON = CACHE / "instances.json"
PHYSICAL_JSON = CACHE / "physical.json"

H = W = 2048

sys.path.insert(0, str(BASE / "src"))
import pq_metric as P  # noqa: E402  (RLE helpers / counts codec)


# --------------------------------------------------------------------------------------
# cached annotation loading
# --------------------------------------------------------------------------------------
def load_coco(path: Path = COCO_JSON) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def cache_paths(cache_dir: Path = CACHE) -> dict[str, Path]:
    return {
        "labels": cache_dir / "labels",
        "instances": cache_dir / "instances.json",
        "physical": cache_dir / "physical.json",
    }


def annotations_by_image(coco: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        out[str(ann["image_id"])].append(ann)
    return out


def _polygon_counts(segmentation, height: int, width: int) -> tuple[np.ndarray, int]:
    """COCO polygon -> absolute RLE run lengths (pycocotools rasterisation) + area."""
    from pycocotools import mask as mask_utils
    if isinstance(segmentation, dict):
        rle = mask_utils.merge([segmentation])
    else:
        polys = segmentation
        if isinstance(polys, list) and polys and isinstance(polys[0], (int, float)):
            polys = [polys]
        rles = mask_utils.frPyObjects(polys, int(height), int(width))
        rle = rles[0] if len(rles) == 1 else mask_utils.merge(rles)
    counts = P.counts_from_compressed(rle["counts"]) if not isinstance(rle["counts"], np.ndarray) else rle["counts"]
    total = int(counts.sum())
    if total != int(height) * int(width):
        # pycocotools normally returns H*W; guard against malformed polygons
        raise ValueError(f"rasterised RLE covers {total} px, expected {height * width}")
    return counts, int(counts[1::2].sum())


def _paint_instances(counts_list: list[np.ndarray], height: int, width: int) -> np.ndarray:
    """Paint RLE run-length lists (column-major, ascending index) into uint16 labels."""
    hw = int(height) * int(width)
    flat = np.zeros(hw, dtype=np.uint16)
    for idx, counts in enumerate(counts_list, start=1):
        pos = 0
        for k in range(counts.size):
            run = int(counts[k])
            if k & 1:
                flat[pos:pos + run] = idx
            pos += run
    return flat.reshape((int(height), int(width)), order="F")


def _union_mask(label: np.ndarray) -> np.ndarray:
    return label > 0


def _semantic_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    if inter == 0:
        return 0.0
    return inter / float(int(np.count_nonzero(a)) + int(np.count_nonzero(b)) - inter)


# --------------------------------------------------------------------------------------
# per-record processing
# --------------------------------------------------------------------------------------
def build_record(image: dict, anns: list[dict], height: int, width: int,
                 labels_dir: Path | None, png_compression: int) -> dict:
    """Rasterise one annotator-image record, write its label PNG, return its metadata."""
    image_id = str(image["id"])
    stem = image_id.split("-", maxsplit=1)[-1]
    annotator_id = image_id.split("-", maxsplit=1)[0]
    anns_sorted = sorted(anns, key=lambda a: str(a["id"]))
    counts_list: list[np.ndarray] = []
    areas: list[int] = []
    for ann in anns_sorted:
        counts, area = _polygon_counts(ann["segmentation"], height, width)
        counts_list.append(counts)
        areas.append(area)
    label = _paint_instances(counts_list, height, width)
    union_area = int(np.count_nonzero(label))
    overlap_px = int(sum(areas) - union_area)
    label_path = ""
    if labels_dir is not None:
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_path = f"labels/{image_id}.png"
        import cv2
        cv2.imwrite(str(labels_dir / f"{image_id}.png"), label,
                    [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)])
    instances = []
    for i, (ann, area) in enumerate(zip(anns_sorted, areas), start=1):
        spine = ann.get("spine")
        spine_flat = [float(v) for v in spine] if isinstance(spine, list) else []
        instances.append({
            "index": i,
            "annotation_id": str(ann["id"]),
            "category_id": int(ann.get("category_id", -1)),
            "area": int(area),                       # rasterised pixel count (== ann["area"])
            "ann_area": float(ann.get("area", -1)),
            "bbox": [float(v) for v in ann.get("bbox", [])],
            "n_spine_points": len(spine_flat) // 2,
            "spine": spine_flat,                     # flat [x1,y1,x2,y2,...] as in the COCO json
        })
    return {
        "image_id": image_id,
        "stem": stem,
        "annotator_id": annotator_id,
        "file_name": str(image.get("file_name", f"{stem}.jpeg")),
        "height": int(height),
        "width": int(width),
        "n_instances": len(instances),
        "union_area": union_area,
        "overlap_px": overlap_px,
        "label_path": label_path,
        "instances": instances,
        "_label": label,        # in-memory only, stripped before json dump
        "_areas": areas,
    }


# --------------------------------------------------------------------------------------
# main build
# --------------------------------------------------------------------------------------
def build(coco_path: Path = COCO_JSON, cache_dir: Path = CACHE, limit: int | None = None,
          write_png: bool = True, png_compression: int = 6,
          verify_sample: int = 20, compare_cv2: int = 0, verbose: bool = True) -> dict:
    t_start = time.time()
    coco = load_coco(coco_path)
    height = int(coco["images"][0]["height"]) if coco["images"] else H
    width = int(coco["images"][0]["width"]) if coco["images"] else W
    paths = cache_paths(cache_dir)
    labels_dir = paths["labels"] if write_png else None
    anns_by_image = annotations_by_image(coco)

    # group image records by physical stem, sorted by annotator id
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for image in coco["images"]:
        by_stem[str(image["id"]).split("-", maxsplit=1)[-1]].append(image)
    stems = sorted(by_stem.keys())
    if limit is not None:
        stems = stems[:int(limit)]
    if verbose:
        print(f"[data_prep] {len(stems)} physical jpegs / "
              f"{sum(len(v) for v in by_stem.values() if True)} annotator-images "
              f"| rasterizer=pycocotools | png={write_png} -> {labels_dir}")

    instances_map: dict[str, dict] = {}
    physical: dict[str, dict] = {}
    n_records = 0
    png_bytes = 0
    t0 = time.time()
    for n_done, stem in enumerate(stems, start=1):
        records = sorted(by_stem[stem], key=lambda im: str(im["id"]))
        rec_meta = []
        for image in records:
            image_id = str(image["id"])
            anns = anns_by_image.get(image_id, [])
            meta = build_record(image, anns, height, width, labels_dir, png_compression)
            if write_png:
                png_path = paths["labels"] / f"{image_id}.png"
                if png_path.exists():
                    png_bytes += png_path.stat().st_size
            label = meta.pop("_label")
            areas = meta.pop("_areas")
            meta["_union"] = _union_mask(label)
            rec_meta.append(meta)
            instances_map[image_id] = {k: v for k, v in meta.items() if not k.startswith("_")}
            n_records += 1
        # ---- medoid annotator: mean pairwise semantic IoU of union masks ----
        n = len(rec_meta)
        pairwise: dict[str, float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                v = _semantic_iou(rec_meta[i]["_union"], rec_meta[j]["_union"])
                pairwise[f"{rec_meta[i]['annotator_id']}|{rec_meta[j]['annotator_id']}"] = round(v, 6)
        mean_iou: dict[str, float] = {}
        for i in range(n):
            vals = [pairwise[f"{min(rec_meta[i]['annotator_id'], rec_meta[j]['annotator_id'])}|"
                             f"{max(rec_meta[i]['annotator_id'], rec_meta[j]['annotator_id'])}"]
                    for j in range(n) if j != i]
            mean_iou[rec_meta[i]["annotator_id"]] = round(float(np.mean(vals)) if vals else 1.0, 6)
        if n == 1:
            medoid_id = rec_meta[0]["annotator_id"]
        else:
            # highest mean IoU, ties -> smallest annotator id
            medoid_id = sorted(mean_iou.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        medoid_rec = next(r for r in rec_meta if r["annotator_id"] == medoid_id)
        physical[stem] = {
            "file_name": str(records[0].get("file_name", f"{stem}.jpeg")),
            "records": [r["image_id"] for r in rec_meta],
            "annotators": [r["annotator_id"] for r in rec_meta],
            "n_records": n,
            "n_instances_total": int(sum(r["n_instances"] for r in rec_meta)),
            "union_areas": {r["annotator_id"]: r["union_area"] for r in rec_meta},
            "pairwise_iou": pairwise,
            "mean_iou": mean_iou,
            "medoid": medoid_id,
            "medoid_record": medoid_rec["image_id"],
            "medoid_label_path": medoid_rec.get("label_path", ""),
        }
        for r in rec_meta:
            r["_union"] = None  # release
        if verbose and n_done % 100 == 0:
            print(f"  [{n_done}/{len(stems)}] {n_records} records, {time.time() - t0:.1f}s")

    # ---------------- global statistics ----------------
    all_areas = [inst["area"] for rec in instances_map.values() for inst in rec["instances"]]
    inst_per_rec = [rec["n_instances"] for rec in instances_map.values()]
    multi = sum(1 for v in physical.values() if v["n_records"] > 1)
    stats = {
        "n_physical_jpegs": len(physical),
        "n_annotator_images": len(instances_map),
        "n_instances": int(sum(inst_per_rec)),
        "n_multi_annotator_stems": int(multi),
        "record_multiplicity": {str(k): sum(1 for v in physical.values() if v["n_records"] == k)
                                for k in sorted({v["n_records"] for v in physical.values()})},
        "instances_per_record": {
            "min": int(min(inst_per_rec)) if inst_per_rec else 0,
            "median": float(np.median(inst_per_rec)) if inst_per_rec else 0.0,
            "max": int(max(inst_per_rec)) if inst_per_rec else 0,
        },
        "instance_area_px": {
            "min": int(min(all_areas)) if all_areas else 0,
            "p25": float(np.percentile(all_areas, 25)) if all_areas else 0.0,
            "median": float(np.median(all_areas)) if all_areas else 0.0,
            "p75": float(np.percentile(all_areas, 75)) if all_areas else 0.0,
            "max": int(max(all_areas)) if all_areas else 0,
            "total": int(sum(all_areas)),
            # mean per-record union area as a fraction of one 2048x2048 image
            "mean_union_fraction_per_record": float(
                np.mean([rec["union_area"] for rec in instances_map.values()]) / (H * W))
                if instances_map else 0.0,
        },
        "category_counts": {str(c): int(cnt) for c, cnt in
                            pd.Series([inst["category_id"] for rec in instances_map.values()
                                       for inst in rec["instances"]]).value_counts().sort_index().items()},
        "spine_points": {
            "total": int(sum(inst["n_spine_points"] for rec in instances_map.values()
                             for inst in rec["instances"])),
            "min": int(min([inst["n_spine_points"] for rec in instances_map.values()
                            for inst in rec["instances"]] or [0])),
            "max": int(max([inst["n_spine_points"] for rec in instances_map.values()
                            for inst in rec["instances"]] or [0])),
        },
        "overlap_px_total": int(sum(rec["overlap_px"] for rec in instances_map.values())),
        "records_with_overlap": int(sum(1 for rec in instances_map.values() if rec["overlap_px"] > 0)),
        "label_png_bytes": int(png_bytes),
        "label_png_mb": round(png_bytes / 1e6, 2),
        "rasterizer": "pycocotools.frPyObjects/mask_utils",
        "label_dtype": "uint16",
        "label_instance_order": "ascending annotation id (UUID string)",
        "build_seconds": round(time.time() - t_start, 1),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # ---------------- validation: label PNG == independent pycocotools decode ----------------
    checks: dict = {}
    if write_png and verify_sample > 0 and instances_map:
        import cv2
        from pycocotools import mask as mask_utils
        sample_ids = list(instances_map.keys())[::max(1, len(instances_map) // verify_sample)][:verify_sample]
        bad = 0
        checked = 0
        for image_id in sample_ids:
            rec = instances_map[image_id]
            img = next(im for im in coco["images"] if str(im["id"]) == image_id)
            anns_sorted = sorted(anns_by_image[image_id], key=lambda a: str(a["id"]))
            rles = []
            for ann in anns_sorted:
                seg = ann["segmentation"]
                rr = mask_utils.frPyObjects(seg if not (isinstance(seg, list) and seg and isinstance(seg[0], (int, float))) else [seg], H, W)
                rles.append(rr[0] if len(rr) == 1 else mask_utils.merge(rr))
            stack = mask_utils.decode(rles)  # (H, W, n)
            expected = np.zeros((H, W), dtype=np.uint16)
            for i in range(len(rles)):
                expected[stack[:, :, i] > 0] = i + 1
            png = cv2.imread(str(paths["labels"] / f"{image_id}.png"), cv2.IMREAD_UNCHANGED)
            checked += 1
            if png is None or png.shape != (H, W) or not np.array_equal(png.astype(np.uint16), expected):
                bad += 1
                if png is not None:
                    mism = int((png.astype(np.uint16) != expected).sum())
                    print(f"    [verify] MISMATCH {image_id}: {mism} px")
        checks["label_png_sample"] = {"checked": checked, "mismatches": int(bad)}
        if verbose:
            print(f"[data_prep] verify {checked} label PNGs against pycocotools decode: "
                  f"{bad} mismatches")
    # every instance rasterised area must equal ann['area'] (proves the rasteriser choice)
    area_mism = 0
    area_checked = 0
    for rec in instances_map.values():
        for inst in rec["instances"]:
            area_checked += 1
            if abs(inst["area"] - inst["ann_area"]) > 1e-6:
                area_mism += 1
    checks["instance_area_equals_coco_area"] = {"checked": area_checked, "mismatches": int(area_mism)}
    if verbose:
        print(f"[data_prep] instance area == ann['area']: {area_checked - area_mism}/{area_checked}")

    # ---------------- optional: quantify cv2.fillPoly vs pycocotools rasterisation ----------------
    if compare_cv2 > 0 and instances_map:
        import cv2
        from pycocotools import mask as mask_utils
        rng = np.random.default_rng(42)
        picked = rng.choice(len(coco["annotations"]), size=min(compare_cv2, len(coco["annotations"])), replace=False)
        ious, ratios = [], []
        for idx in picked:
            ann = coco["annotations"][int(idx)]
            rles = mask_utils.frPyObjects([ann["segmentation"][0]], H, W)
            coco_mask = mask_utils.decode(rles[0]) > 0
            pts = np.asarray(ann["segmentation"][0], dtype=np.float64).reshape(-1, 2)
            canvas = np.zeros((H, W), np.uint8)
            cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
            inter = int(np.count_nonzero(canvas & coco_mask))
            union = int(np.count_nonzero(canvas | coco_mask))
            if union:
                ious.append(inter / union)
                ratios.append(int(canvas.sum()) / max(1.0, float(coco_mask.sum())))
        checks["cv2_vs_pycocotools"] = {
            "n": len(ious),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
            "mean_area_ratio_cv2_over_coco": round(float(np.mean(ratios)), 4) if ratios else None,
        }
        if verbose:
            print(f"[data_prep] cv2.fillPoly vs pycocotools: mean IoU "
                  f"{np.mean(ious):.4f}, area ratio {np.mean(ratios):.4f} (n={len(ious)})")

    # ---------------- write outputs ----------------
    paths["instances"].parent.mkdir(parents=True, exist_ok=True)
    instances_payload = {
        "meta": {"source": str(coco_path), "height": height, "width": width,
                 "rasterizer": stats["rasterizer"], "label_dtype": stats["label_dtype"],
                 "label_instance_order": stats["label_instance_order"],
                 "generated_at": stats["generated_at"]},
        "annotator_images": instances_map,
        "stats": stats,
        "checks": checks,
    }
    paths["instances"].write_text(json.dumps(instances_payload, indent=2), encoding="utf-8")
    physical_payload = {
        "meta": {"source": str(coco_path), "medoid_definition":
                 "record with highest mean pairwise semantic IoU of its instance-union mask "
                 "(tie -> smallest annotator_id); single-record stems: that record"},
        "physical": physical,
    }
    paths["physical"].write_text(json.dumps(physical_payload, indent=2), encoding="utf-8")
    if verbose:
        print(f"[data_prep] instances.json -> {paths['instances']} "
              f"({paths['instances'].stat().st_size / 1e6:.1f} MB)")
        print(f"[data_prep] physical.json  -> {paths['physical']} "
              f"({paths['physical'].stat().st_size / 1e6:.1f} MB)")
        if write_png:
            print(f"[data_prep] labels: {len(instances_map)} PNGs, {png_bytes / 1e6:.1f} MB")
        print(f"[data_prep] done in {time.time() - t_start:.1f}s")
        print(f"[data_prep] stats: {json.dumps({k: v for k, v in stats.items() if k not in ('build_seconds',)}, indent=None)[:900]}")
    return {"stats": stats, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MAGFiLO COCO -> instance label PNG cache")
    ap.add_argument("--coco", default=str(COCO_JSON))
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--limit", type=int, default=None, help="only the first N physical jpegs (smoke test)")
    ap.add_argument("--no-png", action="store_true", help="skip label PNG writing")
    ap.add_argument("--png-compression", type=int, default=6, help="cv2 PNG compression level 0-9")
    ap.add_argument("--verify-sample", type=int, default=20, help="how many label PNGs to re-decode and compare")
    ap.add_argument("--compare-cv2", type=int, default=0, help="compare N random polygons: cv2.fillPoly vs pycocotools")
    args = ap.parse_args(argv)
    out = build(coco_path=Path(args.coco), cache_dir=Path(args.cache), limit=args.limit,
                write_png=not args.no_png, png_compression=args.png_compression,
                verify_sample=args.verify_sample, compare_cv2=args.compare_cv2)
    return 0 if out["stats"]["n_physical_jpegs"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
