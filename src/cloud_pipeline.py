#!/usr/bin/env python3
"""Self-contained Kaggle cloud pipeline for Solar Filament Segmentation 2026.

One script, one run, T4 GPU:
    YOLO11-seg detector (all 1154 annotator records)  ->
    seed-conditioned ResNet18 U-Net crop refiner       ->
    pooled PQ on fold-0 validation (harness official metric) ->
    test inference -> submission.csv

Designed to be uploaded as a Kaggle kernel script (no local imports required).

Inputs
------
* competition data (auto-discovered with rglob):
    /kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/
        train/train_images/*.jpeg
        train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
        test/test_images/*.jpeg
* optional code/config dataset (kaggle-runner uploads src/*.py + work/cache/folds.json):
    /kaggle/input/<code-dataset>/work/cache/folds.json
    /kaggle/input/<code-dataset>/src/pq_metric.py

Outputs (/kaggle/working)
-------------------------
    submission.csv, val_report.json, val_per_image.csv, sweep.csv,
    best_det.pt, best_ref.pt, train_log.txt, config.json, summary.json

Usage
-----
    python cloud_pipeline.py                # full run (30 detector epochs + 8 refiner epochs)
    python cloud_pipeline.py --smoke        # 12 images, 1 epoch each: end-to-end smoke test
    python cloud_pipeline.py --device cpu --smoke --max-physical 2 --img-size 320   # CPU syntax smoke
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ======================================================================================
# 0. Constants / logging
# ======================================================================================
FOLD = 0
DET_EPOCHS = 30
REF_EPOCHS = 8
IMG_SIZE = 1024
BATCH = 8
LABEL_SOURCE = "all"
SMOKE = False

H = W = 2048
NW = 5
SEED = 42

MATCH_IOU = 0.50
MIN_AREA = 48
REFINER_THRESHOLD = 0.65
REFINER_CROP = 256
REFINER_CONTEXT = 1.8
REFINER_MIN_CROP = 96
SEED_DUPLICATE_IOU = 0.60

START_TIME = time.time()
LOG_PATH: Path | None = None


def set_log_path(path: Path) -> None:
    global LOG_PATH
    LOG_PATH = path


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}][+{time.time()-START_TIME:7.1f}s] {message}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# ======================================================================================
# 1. Path discovery
# ======================================================================================
def _candidates_from_roots(patterns: list[str]) -> list[Path]:
    roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd(), Path.cwd().parent]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            try:
                found.extend(sorted(root.glob(pattern)))
            except Exception:  # noqa: BLE001
                continue
    return found


def find_data_root(explicit: str | None = None) -> dict[str, Path]:
    if explicit:
        root = Path(explicit).resolve()
        if (root / "train").exists():
            return _data_paths(root)
    env = os.environ.get("FILAMENT_DATA_ROOT")
    if env and Path(env).exists():
        return _data_paths(Path(env))
    json_candidates: list[Path] = []
    for pattern in (
        "**/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        "**/*Annotations_kaggle2026_train.json",
    ):
        json_candidates.extend(_candidates_from_roots([pattern]))
    if json_candidates:
        root = json_candidates[0].parent.parent
        if (root / "train" / "train_images").exists():
            return _data_paths(root)
    local = Path(__file__).resolve().parent.parent / "data" / "MAGFiLO_1.0_Kaggle_2026"
    if (local / "train").exists():
        log(f"data: using local fallback {local}")
        return _data_paths(local)
    raise FileNotFoundError("could not find MAGFiLO data root (set FILAMENT_DATA_ROOT)")


def _data_paths(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "train_images": root / "train" / "train_images",
        "test_images": root / "test" / "test_images",
        "json": None,
    }
    jsons = sorted((root / "train").glob("*Annotations*kaggle2026_train.json"))
    if not jsons:
        jsons = sorted((root / "train").glob("*train*.json"))
    if not jsons:
        raise FileNotFoundError(f"annotation json not found under {root / 'train'}")
    paths["json"] = jsons[0]
    for key in ("train_images", "test_images"):
        if not paths[key].exists():
            raise FileNotFoundError(paths[key])
    log(f"data: root={root} json={paths['json'].name}")
    return paths


def find_code_root(explicit: str | None = None) -> Path | None:
    if explicit:
        return Path(explicit).resolve()
    env = os.environ.get("FILAMENT_CODE_ROOT")
    if env and Path(env).exists():
        return Path(env)
    markers = _candidates_from_roots(["**/work/cache/folds.json", "**/pq_metric.py", "**/src/pq_metric.py"])
    for marker in markers:
        if marker.name == "folds.json":
            return marker.parent.parent.parent
        return marker.parent if marker.parent.name == "src" else marker.parent
    return None


def find_first(paths: list[Path], name: str) -> Path | None:
    for path in paths:
        if path.name == name and path.exists():
            return path
    return None


# ======================================================================================
# 2. COCO helpers
# ======================================================================================
def load_coco(json_path: Path) -> dict:
    log(f"data: loading COCO json {json_path} ({json_path.stat().st_size/1e6:.1f} MB)")
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_indices(coco: dict, train_images: Path) -> dict:
    from pycocotools import mask as mask_utils

    images_by_id = {str(image["id"]): image for image in coco["images"]}
    annotations_by_image: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[str(annotation["image_id"])].append(annotation)
    image_ids_by_file: dict[str, list[str]] = defaultdict(list)
    for image_id, image in images_by_id.items():
        image_ids_by_file[image["file_name"]].append(image_id)
    for filename in image_ids_by_file:
        image_ids_by_file[filename].sort()
    return {
        "images_by_id": images_by_id,
        "annotations_by_image": dict(annotations_by_image),
        "image_ids_by_file": dict(image_ids_by_file),
        "image_paths": {filename: str(train_images / filename) for filename in image_ids_by_file},
        "mask_utils": mask_utils,
    }


def annotation_to_rle(annotation: dict, indices: dict) -> dict:
    mask_utils = indices["mask_utils"]
    height = width = H
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        return rle
    polygons = segmentation or []
    if not polygons:
        return mask_utils.encode(np.asfortranarray(np.zeros((height, width), dtype=np.uint8)))
    return mask_utils.merge(mask_utils.frPyObjects(polygons, height, width))


# ======================================================================================
# 3. Folds (harness work/cache/folds.json, deterministic grouped fallback)
# ======================================================================================
def _fold_map_from_payload(payload) -> dict[str, int] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("fold_of"), dict):
        return {str(k): int(v) for k, v in payload["fold_of"].items()}
    if isinstance(payload.get("folds"), dict):
        result = {}
        for fold, files in payload["folds"].items():
            if not isinstance(files, list):
                return None
            for filename in files:
                result[str(filename)] = int(fold)
        return result
    if payload and all(isinstance(value, int) for value in payload.values()):
        return {str(key): int(value) for key, value in payload.items()}
    return None


def load_folds(filenames: list[str], code_root: Path | None, folds_override: str | None) -> tuple[dict[str, int], str]:
    candidates: list[Path] = []
    if folds_override:
        candidates.append(Path(folds_override))
    if code_root is not None:
        candidates.append(code_root / "work" / "cache" / "folds.json")
        candidates.extend(code_root.glob("**/work/cache/folds.json"))
    candidates.extend(_candidates_from_roots(["**/folds.json"]))
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            folds = _fold_map_from_payload(payload)
        except Exception as error:  # noqa: BLE001
            log(f"folds: failed to read {path} ({error})")
            continue
        if not folds:
            continue
        aligned = {}
        ok = True
        for filename in filenames:
            stem = Path(filename).stem
            if filename in folds:
                aligned[filename] = int(folds[filename])
            elif stem in folds:
                aligned[filename] = int(folds[stem])
            else:
                ok = False
                break
        if ok:
            log(f"folds: using {path} ({len(aligned)} physical images)")
            return aligned, str(path)
        log(f"folds: {path} does not cover all images, skipped")
    files = np.array(sorted(filenames))
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(len(files))
    assignment = np.empty(len(files), dtype=int)
    assignment[permutation] = np.arange(len(files)) % NW
    folds = {str(filename): int(fold) for filename, fold in zip(files, assignment)}
    log("folds: WARNING harness folds.json not found -> deterministic grouped split seed=42")
    return folds, "fallback_seed42"


# ======================================================================================
# 4. YOLO dataset (one sample per annotation record, images pre-resized)
# ======================================================================================
def sample_stem_for(image_id: str, filename: str) -> str:
    annotator = image_id.split("-", 1)[0]
    return f"{Path(filename).stem}__{annotator}"


def build_image_cache(indices: dict, resize: int, out_dir: Path, limit: int = 0) -> dict[str, dict]:
    meta_path = out_dir / "images" / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, dict] = {}
    filenames = sorted(indices["image_ids_by_file"])
    if limit:
        filenames = filenames[:limit]
    started = time.time()
    for number, filename in enumerate(filenames, 1):
        image = cv2.imread(indices["image_paths"][filename], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(indices["image_paths"][filename])
        if image.shape[0] != resize or image.shape[1] != resize:
            image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_AREA)
        for image_id in indices["image_ids_by_file"][filename]:
            stem = sample_stem_for(image_id, filename)
            destination = image_dir / f"{stem}.jpg"
            if not destination.exists():
                cv2.imwrite(str(destination), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            mapping[stem] = {"filename": filename, "image_id": image_id, "path": str(destination)}
        if number % 100 == 0:
            log(f"image cache: {number}/{len(filenames)} physical images")
    meta_path.write_text(json.dumps(mapping), encoding="utf-8")
    log(f"image cache: {len(mapping)} samples at {resize}px in {time.time()-started:.1f}s")
    return mapping


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def annotation_to_yolo_line(annotation: dict) -> str | None:
    segmentation = annotation.get("segmentation") or []
    if not isinstance(segmentation, list) or not segmentation:
        return None
    polygons = []
    for polygon in segmentation:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) >= 3:
            polygons.append(points)
    if not polygons:
        return None
    polygon = max(polygons, key=polygon_area)
    normalized = np.clip(polygon / float(H), 0.0, 1.0)
    return "0 " + " ".join(f"{value:.6f}" for value in normalized.ravel())


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_split(split_dir: Path, filenames: list[str], all_images: dict[str, dict], indices: dict) -> tuple[int, int]:
    wanted = set(filenames)
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    n_images = 0
    n_boxes = 0
    for stem, sample in all_images.items():
        if sample["filename"] not in wanted:
            continue
        link_or_copy(Path(sample["path"]), image_dir / f"{stem}.jpg")
        lines = []
        for annotation in indices["annotations_by_image"].get(sample["image_id"], []):
            if annotation.get("iscrowd", 0):
                continue
            line = annotation_to_yolo_line(annotation)
            if line is not None:
                lines.append(line)
        n_boxes += len(lines)
        (label_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_images += 1
    return n_images, n_boxes


def build_yolo_dataset(args, indices: dict, folds: dict[str, int], out_dir: Path, train_files, val_files) -> Path:
    resize = args.img_size
    if args.smoke:
        resize = args.img_size
    all_images = build_image_cache(indices, resize, out_dir)
    fold_dir = out_dir / "yolo_data"
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    yaml_path = fold_dir / "data.yaml"
    started = time.time()
    n_train, b_train = write_split(fold_dir / "train", train_files, all_images, indices)
    n_val, b_val = write_split(fold_dir / "val", val_files, all_images, indices)
    yaml_path.write_text(
        f"path: {fold_dir.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n  0: filament\n",
        encoding="utf-8",
    )
    log(
        f"yolo dataset: train {n_train} samples / {b_train} boxes, val {n_val} samples / "
        f"{b_val} boxes in {time.time()-started:.1f}s"
    )
    return yaml_path


# ======================================================================================
# 5. Detector
# ======================================================================================
def _import_yolo():
    """Import ultralytics, installing it once if the Kaggle image lacks it."""
    try:
        from ultralytics import YOLO

        return YOLO
    except ImportError:
        log("detector: ultralytics missing -> pip install ultralytics")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
        from ultralytics import YOLO

        return YOLO


def train_detector(args, yaml_path: Path, out_dir: Path, train_count: int) -> Path:
    import torch

    YOLO = _import_yolo()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log("detector: CUDA not available -> CPU")
        device = "cpu"
    model_name = args.det_model
    if args.smoke and args.det_model == "yolo11s-seg.pt":
        model_name = "yolo11n-seg.pt"
        log("detector: SMOKE -> yolo11n-seg.pt")
    model = YOLO(model_name)
    epochs = args.det_epochs
    batch = args.batch
    imgsz = args.img_size
    if args.smoke:
        epochs = min(epochs, 1)
        batch = min(batch, 4)
    started = time.time()
    try:
        model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=0 if device == "cuda" else "cpu",
            workers=args.workers,
            seed=SEED,
            patience=max(5, epochs // 4),
            project=str(out_dir / "runs"),
            name="detector",
            exist_ok=True,
            pretrained=True,
            single_cls=True,
            mask_ratio=2,
            overlap_mask=False,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.25,
            fliplr=0.5,
            flipud=0.5,
            degrees=0.0,
            scale=0.20,
            translate=0.05,
            mosaic=0.0,
            close_mosaic=0,
            cache=False,
            amp=device == "cuda",
            plots=False,
            val=not args.smoke and not args.no_val,
            verbose=True,
        )
    except RuntimeError as error:
        log(f"detector: training raised {error}")
        raise
    weights_dir = out_dir / "runs" / "detector" / "weights"
    best = out_dir / "best_det.pt"
    source = weights_dir / "best.pt"
    if not source.exists():
        source = weights_dir / "last.pt"
    if not source.exists():
        raise FileNotFoundError(f"detector weights missing under {weights_dir}")
    shutil.copy2(source, best)
    log(f"detector: {epochs} epochs in {(time.time()-started)/60:.1f} min -> {best}")
    return best


def load_detector(weights: Path, args):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    log(f"detector: loaded {weights}")
    return model


# ======================================================================================
# 6. Refiner: crop cache + training
# ======================================================================================
def annotation_to_mask(annotation: dict, indices: dict) -> np.ndarray:
    return indices["mask_utils"].decode(annotation_to_rle(annotation, indices)).astype(bool)


def square_bounds_from_mask(mask: np.ndarray, context: float = REFINER_CONTEXT, minimum: int = REFINER_MIN_CROP):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, width, height
    center_x = (xs.min() + xs.max() + 1) / 2.0
    center_y = (ys.min() + ys.max() + 1) / 2.0
    side = int(math.ceil(max(np.ptp(xs) + 1, np.ptp(ys) + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return x0, y0, x1, y1


def build_refiner_cache(files: list[str], indices: dict, crop: int, label_source: str, out_dir: Path, split: str) -> tuple[Path, Path]:
    image_path = out_dir / f"refiner_{split}_images.npy"
    target_path = out_dir / f"refiner_{split}_targets.npy"
    meta_path = out_dir / f"refiner_{split}_meta.json"
    if meta_path.exists() and image_path.exists():
        return image_path, target_path
    samples = []
    for filename in files:
        image_ids = indices["image_ids_by_file"][filename]
        if label_source == "medoid":
            image_ids = [image_ids[0]]
        for image_id in image_ids:
            for annotation in indices["annotations_by_image"].get(image_id, []):
                if annotation.get("iscrowd", 0) or not annotation.get("segmentation"):
                    continue
                samples.append((filename, image_id, annotation))
    started = time.time()
    images = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.uint8, shape=(len(samples), crop, crop))
    targets = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.uint8, shape=(len(samples), crop, crop))
    cursor = 0
    cached_image: np.ndarray | None = None
    cached_filename = None
    for filename, image_id, annotation in samples:
        if filename != cached_filename:
            cached_image = cv2.imread(indices["image_paths"][filename], cv2.IMREAD_GRAYSCALE)
            if cached_image is None:
                raise FileNotFoundError(indices["image_paths"][filename])
            cached_filename = filename
        mask = annotation_to_mask(annotation, indices)
        if not mask.any():
            continue
        x0, y0, x1, y1 = square_bounds_from_mask(mask)
        images[cursor] = cv2.resize(cached_image[y0:y1, x0:x1], (crop, crop), interpolation=cv2.INTER_LINEAR)
        targets[cursor] = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8), (crop, crop), interpolation=cv2.INTER_NEAREST)
        cursor += 1
    images.flush()
    targets.flush()
    del images, targets
    meta_path.write_text(
        json.dumps({"split": split, "crop": crop, "label_source": label_source, "n_samples": cursor}), encoding="utf-8"
    )
    log(f"refiner cache {split}: {cursor} crops in {time.time()-started:.1f}s -> {image_path}")
    return image_path, target_path


def build_refiner_class():
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    class ConvBlock2(nn.Sequential):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

    class SeedRefiner(nn.Module):
        def __init__(self, pretrained: bool = True) -> None:
            super().__init__()
            encoder = None
            if pretrained:
                try:
                    encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
                except Exception as error:  # noqa: BLE001 - offline Kaggle kernels
                    log(f"refiner: resnet18 ImageNet weights unavailable ({error}) -> random init")
                    encoder = resnet18(weights=None)
            else:
                encoder = resnet18(weights=None)
            self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
            self.pool = encoder.maxpool
            self.layer1 = encoder.layer1
            self.layer2 = encoder.layer2
            self.layer3 = encoder.layer3
            self.layer4 = encoder.layer4
            self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
            self.dec4 = ConvBlock2(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
            self.dec3 = ConvBlock2(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
            self.dec2 = ConvBlock2(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
            self.dec1 = ConvBlock2(96, 32)
            self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
            self.head = nn.Conv2d(16, 1, 1)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, tensor):
            tensor = (tensor - self.mean) / self.std
            x0 = self.stem(tensor)
            x1 = self.layer1(self.pool(x0))
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            value = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
            value = self.dec3(torch.cat([self.up3(value), x2], dim=1))
            value = self.dec2(torch.cat([self.up2(value), x1], dim=1))
            value = self.dec1(torch.cat([self.up1(value), x0], dim=1))
            return self.head(self.up0(value))

    return SeedRefiner


def _erode_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(radius) + 1, 2 * abs(radius) + 1))
    return cv2.dilate(mask, kernel) if radius > 0 else cv2.erode(mask, kernel)


def coarsen_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    small = cv2.resize(mask, (max(1, mask.shape[1] // factor), max(1, mask.shape[0] // factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)


def box_seed(target: np.ndarray, pad_ratio: float) -> np.ndarray:
    ys, xs = np.where(target > 0)
    seed = np.zeros_like(target)
    if len(xs) == 0:
        return seed
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    pad_x = int(round((x1 - x0) * pad_ratio))
    pad_y = int(round((y1 - y0) * pad_ratio))
    seed[max(0, y0 - pad_y) : min(seed.shape[0], y1 + pad_y), max(0, x0 - pad_x) : min(seed.shape[1], x1 + pad_x)] = 1
    return seed


def synthesise_seed(target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mode = rng.choice(["mask", "mask", "mask", "coarse", "box"])
    target_bool = target > 0
    if mode == "box":
        return box_seed(target_bool, float(rng.uniform(0.15, 0.6))).astype(np.uint8)
    seed = target_bool.astype(np.uint8)
    seed = _erode_dilate(seed, int(rng.integers(-6, 7)))
    if mode == "coarse":
        seed = coarsen_mask(seed, int(rng.integers(2, 5)))
    if rng.random() < 0.7:
        limit = max(4, int(0.12 * target.shape[0]))
        matrix = np.float32([[1, 0, int(rng.integers(-limit, limit + 1))], [0, 1, int(rng.integers(-limit, limit + 1))]])
        seed = cv2.warpAffine(seed, matrix, (seed.shape[1], seed.shape[0]), flags=cv2.INTER_NEAREST)
    if seed.sum() == 0:
        seed = target_bool.astype(np.uint8)
    return seed


class RefinerDataset:
    def __init__(self, images_path: Path, targets_path: Path, crop: int, augment: bool, seed: int = SEED):
        self.images = np.load(images_path, mmap_mode="r")
        self.targets = np.load(targets_path, mmap_mode="r")
        # crop=None/<=0 -> use the actual cached crop side (single source of truth)
        self.crop = int(crop) if crop is not None and int(crop) > 0 else int(self.images.shape[1])
        self.augment = augment
        self.seed = seed
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch

        image = np.asarray(self.images[index], dtype=np.uint8)
        target = np.asarray(self.targets[index], dtype=np.uint8)
        clahe = self._clahe.apply(image)
        rng = np.random.default_rng(self.seed + index + random.randint(0, 1_000_000))
        if self.augment and rng.random() < 0.5:
            seed_crop = coarsen_mask(target, int(rng.integers(2, 5)))
            seed_crop = _erode_dilate(seed_crop, int(rng.integers(-4, 5)))
        else:
            seed_crop = synthesise_seed(target, rng) if self.augment else target
        features = np.stack([image, clahe, seed_crop * 255], axis=0).astype(np.float32) / 255.0
        if self.augment:
            k = random.randrange(4)
            features = np.rot90(features, k, axes=(1, 2)).copy()
            target = np.rot90(target, k).copy()
            if random.random() < 0.5:
                features = features[:, :, ::-1].copy()
                target = target[:, ::-1].copy()
            zoom = random.uniform(0.85, 1.15)
            if abs(zoom - 1.0) > 0.02:
                side = max(8, int(round(self.crop * zoom)))
                resized_features = np.stack(
                    [cv2.resize(features[channel], (side, side), interpolation=cv2.INTER_LINEAR) for channel in range(3)]
                )
                resized_target = cv2.resize(target, (side, side), interpolation=cv2.INTER_NEAREST)
                if side >= self.crop:
                    offset = (side - self.crop) // 2
                    resized_features = resized_features[:, offset : offset + self.crop, offset : offset + self.crop]
                    resized_target = resized_target[offset : offset + self.crop, offset : offset + self.crop]
                else:
                    pad = self.crop - side
                    top, left = pad // 2, pad // 2
                    resized_features = np.pad(
                        resized_features, ((0, 0), (top, pad - top), (left, pad - left)), mode="reflect"
                    )
                    resized_target = np.pad(resized_target, ((top, pad - top), (left, pad - left)), mode="constant")
                features = resized_features
                target = resized_target
            gain = random.uniform(0.9, 1.1)
            features[:2] = np.clip(features[:2] * gain, 0.0, 1.0)
        return torch.from_numpy(features), torch.from_numpy(target[None].astype(np.float32))


def soft_dice_loss(logits, targets):
    import torch

    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def train_refiner(args, train_paths, val_paths, out_dir: Path, ref_crop: int) -> Path:
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    device = torch.device("cuda:0" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    SeedRefiner = build_refiner_class()
    model = SeedRefiner(pretrained=True).to(device)

    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, dataset: RefinerDataset) -> None:
            self.dataset = dataset

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, index: int):
            return self.dataset[index]

    train_dataset = TorchDataset(RefinerDataset(train_paths[0], train_paths[1], crop=-1, augment=True))
    val_dataset = TorchDataset(RefinerDataset(val_paths[0], val_paths[1], crop=-1, augment=False))
    train_loader = DataLoader(train_dataset, batch_size=args.ref_batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.ref_batch * 2, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs = 1 if args.smoke else args.ref_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_dice = -1.0
    checkpoint = out_dir / "best_ref.pt"
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(features)
                loss = 0.45 * functional.binary_cross_entropy_with_logits(logits, targets) + 0.55 * soft_dice_loss(
                    logits, targets
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        scheduler.step()
        model.eval()
        dice_total = 0.0
        count = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                logits = model(features)
                predictions = logits.sigmoid() >= REFINER_THRESHOLD
                target_bool = targets > 0.5
                intersection = (predictions & target_bool).sum(dim=(1, 2, 3)).float()
                denominator = predictions.sum(dim=(1, 2, 3)) + target_bool.sum(dim=(1, 2, 3))
                dice_total += float(((2 * intersection + 1e-7) / (denominator + 1e-7)).sum().item())
                count += len(features)
        val_dice = dice_total / max(count, 1)
        log(f"refiner: epoch {epoch}/{epochs} train_loss={np.mean(losses):.4f} val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": best_dice}, checkpoint)
    if not checkpoint.exists():
        torch.save({"model": model.state_dict(), "epoch": epochs, "val_dice": best_dice}, checkpoint)
    log(f"refiner: {epochs} epochs in {(time.time()-started)/60:.1f} min, best val dice={best_dice:.4f} -> {checkpoint}")
    return checkpoint


# ======================================================================================
# 7. Inference: seeds -> refiner probability -> entries
# ======================================================================================
def entry_from_full(mask: np.ndarray) -> dict | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    return {"x0": x0, "y0": y0, "m": np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=np.uint8)}


def entry_area(entry: dict) -> int:
    return int(entry["m"].sum())


def entry_to_full(entry: dict) -> np.ndarray:
    full = np.zeros((H, W), dtype=bool)
    full[entry["y0"] : entry["y0"] + entry["m"].shape[0], entry["x0"] : entry["x0"] + entry["m"].shape[1]] = entry["m"] > 0
    return full


def entries_iou(first: dict, second: dict) -> float:
    x0 = max(first["x0"], second["x0"])
    y0 = max(first["y0"], second["y0"])
    x1 = min(first["x0"] + first["m"].shape[1], second["x0"] + second["m"].shape[1])
    y1 = min(first["y0"] + first["m"].shape[0], second["y0"] + second["m"].shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    a = first["m"][y0 - first["y0"] : y1 - first["y0"], x0 - first["x0"] : x1 - first["x0"]]
    b = second["m"][y0 - second["y0"] : y1 - second["y0"], x0 - second["x0"] : x1 - second["x0"]]
    intersection = int(np.logical_and(a, b).sum())
    if intersection == 0:
        return 0.0
    union = entry_area(first) + entry_area(second) - intersection
    return float(intersection / union) if union > 0 else 0.0


def subtract_entry(target: dict, other: dict) -> dict | None:
    x0 = max(target["x0"], other["x0"])
    y0 = max(target["y0"], other["y0"])
    x1 = min(target["x0"] + target["m"].shape[1], other["x0"] + other["m"].shape[1])
    y1 = min(target["y0"] + target["m"].shape[0], other["y0"] + other["m"].shape[0])
    if x1 <= x0 or y1 <= y0:
        return target
    view = target["m"][y0 - target["y0"] : y1 - target["y0"], x0 - target["x0"] : x1 - target["x0"]]
    other_view = other["m"][y0 - other["y0"] : y1 - other["y0"], x0 - other["x0"] : x1 - other["x0"]]
    if not np.any(np.logical_and(view, other_view)):
        return target
    updated = target["m"].copy()
    updated[y0 - target["y0"] : y1 - target["y0"], x0 - target["x0"] : x1 - target["x0"]][
        np.logical_and(view, other_view)
    ] = 0
    ys, xs = np.where(updated)
    if len(xs) == 0:
        return None
    by0, by1 = int(ys.min()), int(ys.max()) + 1
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    return {
        "x0": target["x0"] + bx0,
        "y0": target["y0"] + by0,
        "m": np.ascontiguousarray(updated[by0:by1, bx0:bx1], dtype=np.uint8),
    }


def resolve_entries(entries: list[dict], scores: list[float], minimum_area: int) -> list[dict]:
    kept: list[dict] = []
    kept_scores: list[float] = []
    for index in np.argsort(scores)[::-1]:
        entry = entries[int(index)]
        if entry_area(entry) < minimum_area:
            continue
        updated = entry
        for previous in kept:
            updated = subtract_entry(updated, previous)
            if updated is None:
                break
        if updated is None or entry_area(updated) < minimum_area:
            continue
        kept.append(updated)
        kept_scores.append(float(scores[int(index)]))
    return kept


def detect_seeds(detector, image_bgr: np.ndarray, device: str, conf: float, imgsz: int = IMG_SIZE) -> list[dict]:
    result = detector.predict(
        source=image_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=0.5,
        agnostic_nms=True,
        retina_masks=False,
        verbose=False,
        device=device,
        max_det=300,
    )[0]
    if result.masks is None or result.boxes is None or len(result.masks.data) == 0:
        return []
    masks = result.masks.data.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy().astype(float)
    seeds = []
    for mask, score in zip(masks, scores):
        mask = mask.astype(np.uint8)
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        if int(mask.sum()) < MIN_AREA:
            continue
        seeds.append({"mask": mask.astype(bool), "score": float(score)})
    kept: list[dict] = []
    for seed in sorted(seeds, key=lambda item: -item["score"]):
        entry = entry_from_full(seed["mask"])
        if entry is None:
            continue
        if all(entries_iou(entry, other["entry"]) < SEED_DUPLICATE_IOU for other in kept):
            seed["entry"] = entry
            kept.append(seed)
    return kept


def choose_probability(prob_base: np.ndarray, prob_tta: np.ndarray, threshold: float, tta_mode: str) -> np.ndarray:
    if tta_mode == "none":
        return prob_base
    base_mask = prob_base >= threshold
    candidate_mask = prob_tta >= threshold
    use_candidate = True
    if base_mask.any():
        area_ratio = candidate_mask.sum() / max(base_mask.sum(), 1)
        intersection = np.logical_and(base_mask, candidate_mask).sum()
        dice = float((2 * intersection + 1e-7) / (base_mask.sum() + candidate_mask.sum() + 1e-7))
        components_base = max(cv2.connectedComponents(base_mask.astype(np.uint8), connectivity=8)[0] - 1, 0)
        components_candidate = max(cv2.connectedComponents(candidate_mask.astype(np.uint8), connectivity=8)[0] - 1, 0)
        use_candidate = 0.60 <= area_ratio <= 1.70 and dice >= 0.62 and components_candidate <= components_base + 1
    return prob_tta if use_candidate else prob_base


def entries_for_config(refined: list[dict], conf_threshold: float, ref_threshold: float, tta_mode: str):
    entries = []
    scores = []
    for item in refined:
        if item["score"] < conf_threshold:
            continue
        probability = choose_probability(item["prob_base"], item["prob_tta"], ref_threshold, tta_mode)
        x0, y0, x1, y1 = item["bounds"]
        crop_mask = cv2.resize(
            (probability >= ref_threshold).astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        if not crop_mask.any():
            crop_mask = cv2.resize(
                item["seed_crop"].astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        entry = entry_from_full(np.pad(crop_mask, ((y0, H - y1), (x0, W - x1)), mode="constant"))
        if entry is None:
            continue
        confidence = float(probability[probability >= ref_threshold].mean()) if (probability >= ref_threshold).any() else 0.0
        entries.append(entry)
        scores.append(float(item["score"] * confidence))
    return resolve_entries(entries, scores, MIN_AREA)


def predict_image(
    image_path: Path,
    detector,
    refiner,
    device: str,
    conf: float,
    args,
    imgsz: int | None = None,
    ref_crop: int | None = None,
) -> list[dict]:
    import torch

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    if image.shape != (H, W):
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    seeds = detect_seeds(
        detector, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), device, conf, imgsz=(imgsz or args.img_size)
    )
    if not seeds:
        return []
    crop = ref_crop or getattr(args, "ref_crop", 256)
    prepared = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for seed in seeds:
        x0, y0, x1, y1 = square_bounds_from_mask(seed["mask"])
        image_crop = cv2.resize(image[y0:y1, x0:x1], (crop, crop), interpolation=cv2.INTER_LINEAR)
        seed_crop = cv2.resize(seed["mask"][y0:y1, x0:x1].astype(np.uint8), (crop, crop), interpolation=cv2.INTER_NEAREST)
        # NOTE: SeedRefiner.forward normalises internally -> feed raw [0,1] features only.
        features = np.stack([image_crop, clahe.apply(image_crop), seed_crop * 255], axis=0).astype(np.float32) / 255.0
        prepared.append(
            {
                "features": features,
                "bounds": (x0, y0, x1, y1),
                "score": seed["score"],
                "seed_crop": seed_crop.astype(bool),
            }
        )
    with torch.no_grad():
        for start in range(0, len(prepared), 8):
            chunk = prepared[start : start + 8]
            batch = torch.from_numpy(np.stack([item["features"] for item in chunk])).to(device)
            base = torch.sigmoid(refiner(batch))
            accumulator = base.clone()
            for dims in ((-1,), (-2,), (-2, -1)):
                accumulator += torch.flip(torch.sigmoid(refiner(torch.flip(batch, dims=dims))), dims=dims)
            base_np = base[:, 0].detach().cpu().numpy().astype(np.float32)
            tta_np = (accumulator / 4.0)[:, 0].detach().cpu().numpy().astype(np.float32)
            for offset, item in enumerate(chunk):
                item["prob_base"] = base_np[offset]
                item["prob_tta"] = tta_np[offset]
    return prepared


# ======================================================================================
# 8. PQ evaluation (harness official metric first, embedded fallback second)
# ======================================================================================
def import_harness_metric(code_root: Path | None) -> tuple[object | None, str]:
    candidates: list[Path] = []
    if code_root is not None:
        candidates.extend(code_root.glob("**/pq_metric.py"))
        candidates.append(code_root / "src" / "pq_metric.py")
    candidates.extend(_candidates_from_roots(["**/pq_metric.py"]))
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        try:
            spec = importlib.util.spec_from_file_location("pq_metric_harness", path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            if hasattr(module, "evaluate") and hasattr(module, "gt_df_from_coco"):
                log(f"pq: imported harness metric {path}")
                return module, str(path)
        except Exception as error:  # noqa: BLE001
            log(f"pq: failed to import {path} ({error})")
    return None, "embedded"


def pq_embedded(gt_sets: list[list[dict]], pred_entries: list[dict]) -> dict:
    """Official PQ formula on bbox-local entries (exact integer intersections)."""
    tp = 0
    sum_iou = 0.0
    fp = 0
    fn = 0
    for gt_entries in gt_sets:
        n_gt = len(gt_entries)
        n_pred = len(pred_entries)
        if n_gt == 0 and n_pred == 0:
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        if n_gt == 0:
            fp += n_pred
            continue
        matched_pred = [False] * n_pred
        matched_gt = [False] * n_gt
        for gt_index, gt_entry in enumerate(gt_entries):
            for pred_index, pred_entry in enumerate(pred_entries):
                iou = entries_iou(gt_entry, pred_entry)
                if iou > MATCH_IOU:
                    tp += 1
                    sum_iou += iou
                    matched_gt[gt_index] = True
                    matched_pred[pred_index] = True
        fp += matched_pred.count(False)
        fn += matched_gt.count(False)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": sum_iou / denominator if denominator else 0.0,
        "sq": sum_iou / tp if tp else 0.0,
        "rq": tp / denominator if denominator else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sum_tp_iou": sum_iou,
        "denominator": denominator,
    }


def gt_entries_for_stem(filename: str, indices: dict) -> list[list[dict]]:
    sets = []
    for image_id in indices["image_ids_by_file"][filename]:
        entries = []
        for annotation in indices["annotations_by_image"].get(image_id, []):
            if annotation.get("iscrowd", 0):
                continue
            mask = annotation_to_mask(annotation, indices)
            entry = entry_from_full(mask)
            if entry is not None:
                entries.append(entry)
        sets.append(entries)
    return sets


def entries_to_pred_rows(stem: str, entries: list[dict]) -> list[dict]:
    import pycocotools.mask as mask_utils

    rows = []
    for index, entry in enumerate(entries, 1):
        counts = mask_utils.encode(np.asfortranarray(entry_to_full(entry).astype(np.uint8)))["counts"]
        rows.append(
            {"filament_id": f"{stem}_{index}", "segmentation_rle": counts.decode("ascii") if isinstance(counts, bytes) else str(counts)}
        )
    return rows


# ======================================================================================
# 9. Main pipeline
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--det-epochs", type=int, default=DET_EPOCHS)
    parser.add_argument("--ref-epochs", type=int, default=REF_EPOCHS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--ref-batch", type=int, default=16)
    parser.add_argument("--label-source", default=LABEL_SOURCE, choices=["all", "medoid"])
    parser.add_argument("--det-model", default="yolo11s-seg.pt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-physical", type=int, default=0, help="debug: cap number of physical train images")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--code-root", default=None)
    parser.add_argument("--folds-json", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--det-weights", default=None, help="skip detector training and use these weights")
    parser.add_argument("--refiner-weights", default=None, help="skip refiner training and use these weights")
    parser.add_argument("--no-val", action="store_true", help="detector training without per-epoch validation")
    parser.add_argument("--no-sweep", action="store_true", help="skip threshold sweep")
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    return parser


def out_dir_for(args) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return Path(__file__).resolve().parent.parent / "work" / "cloud_export"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        log("main: CUDA requested but not available -> CPU")
        args.device = "cpu"

    out_dir = out_dir_for(args).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_log_path(out_dir / "train_log.txt")
    if args.smoke:
        args.det_epochs = 1
        args.ref_epochs = 1
        args.img_size = min(args.img_size, 512)
        args.batch = min(args.batch, 2)
        args.ref_batch = min(args.ref_batch, 4)
        args.workers = min(args.workers, 2)
    log("=" * 100)
    log(f"cloud_pipeline start: smoke={args.smoke} args={vars(args)}")
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={args.device}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")

    config = dict(vars(args))
    config["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["git_script"] = str(Path(__file__).resolve())
    (out_dir / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")

    timings: dict[str, float] = {}
    stage_started = time.time()

    # ---------------- data ----------------
    data = find_data_root(args.data_root)
    code_root = find_code_root(args.code_root)
    log(f"code_root: {code_root}")
    coco = load_coco(data["json"])
    indices = build_indices(coco, data["train_images"])
    filenames = sorted(indices["image_ids_by_file"])
    folds, fold_source = load_folds(filenames, code_root, args.folds_json)
    fold = args.fold
    train_files = [filename for filename in filenames if folds[filename] != fold]
    val_files = [filename for filename in filenames if folds[filename] == fold]
    if args.max_physical:
        train_files = train_files[: args.max_physical]
        val_files = val_files[: max(2, args.max_physical // 3)]
    if args.smoke:
        smoke_train = train_files[:10]
        smoke_val = val_files[:4]
        if not smoke_val:
            smoke_val = train_files[-2:]
            smoke_train = smoke_train[:-2]
        train_files, val_files = smoke_train, smoke_val
        args.test_limit = args.test_limit or 4
    n_train_records = sum(len(indices["image_ids_by_file"][f]) for f in train_files)
    log(
        f"data: {len(filenames)} physical images | fold {fold}: train {len(train_files)} physical/"
        f"{n_train_records} records, val {len(val_files)} physical; folds={fold_source}"
    )
    (out_dir / "folds_used.json").write_text(
        json.dumps({"source": fold_source, "fold": fold, "fold_of": folds}, indent=1), encoding="utf-8"
    )
    timings["data"] = time.time() - stage_started
    stage_started = time.time()

    # ---------------- detector ----------------
    detector_weights = Path(args.det_weights) if args.det_weights else None
    if detector_weights is not None and detector_weights.exists():
        log(f"detector: skipping training, using {detector_weights}")
    else:
        yaml_path = build_yolo_dataset(args, indices, folds, out_dir, train_files, val_files)
        detector_weights = train_detector(args, yaml_path, out_dir, len(train_files))
    timings["detector"] = time.time() - stage_started
    stage_started = time.time()

    # ---------------- refiner ----------------
    ref_crop = 128 if args.smoke else 256
    args.ref_crop = ref_crop
    refiner_weights = Path(args.refiner_weights) if args.refiner_weights else None
    if refiner_weights is not None and refiner_weights.exists():
        log(f"refiner: skipping training, using {refiner_weights}")
    else:
        if args.smoke:
            train_ref_files = train_files[:4]
            val_ref_files = val_files[:2]
        else:
            train_ref_files, val_ref_files = train_files, val_files
        train_paths = build_refiner_cache(train_ref_files, indices, ref_crop, args.label_source, out_dir, "train")
        val_paths = build_refiner_cache(val_ref_files, indices, ref_crop, args.label_source, out_dir, "val")
        args.ref_crop = ref_crop
        refiner_weights = train_refiner(args, train_paths, val_paths, out_dir, ref_crop)
    log(f"refiner: trained at crop={ref_crop}")
    timings["refiner"] = time.time() - stage_started
    stage_started = time.time()

    # ---------------- models ----------------
    detector = load_detector(detector_weights, args)
    SeedRefiner = build_refiner_class()
    refiner_payload = torch.load(refiner_weights, map_location="cpu", weights_only=False)
    refiner_state = refiner_payload["model"] if isinstance(refiner_payload, dict) and "model" in refiner_payload else refiner_payload
    refiner = SeedRefiner(pretrained=False)
    refiner.load_state_dict(refiner_state)
    device = "cuda:0" if args.device == "cuda" else "cpu"
    refiner = refiner.to(device).eval()
    log(f"models: detector={detector_weights} refiner={refiner_weights} device={device}")

    # ---------------- val inference + sweep ----------------
    val_stems = [Path(filename).stem for filename in val_files]
    val_gt_sets: dict[str, list[list[dict]]] = {}
    refined_cache: dict[str, list[dict]] = {}
    for number, filename in enumerate(val_files, 1):
        stem = Path(filename).stem
        val_gt_sets[stem] = gt_entries_for_stem(filename, indices)
        refined_cache[stem] = predict_image(
            Path(indices["image_paths"][filename]),
            detector,
            refiner,
            device,
            0.05,
            args,
            imgsz=args.img_size,
            ref_crop=ref_crop,
        )
        if number % 20 == 0:
            log(f"val inference: {number}/{len(val_files)} images")
    if args.val_limit:
        val_stems = val_stems[: args.val_limit]
    log(f"val inference done in {time.time()-stage_started:.1f}s")

    conf_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35] if not args.smoke else [0.10, 0.20]
    ref_grid = [0.50, 0.60, 0.65, 0.70, 0.75] if not args.smoke else [0.65]
    tta_grid = ["none", "guard"] if not args.smoke else ["guard"]
    sweep_rows = []
    if not args.no_sweep:
        for conf_threshold in conf_grid:
            for ref_threshold in ref_grid:
                for tta_mode in tta_grid:
                    tp = sum_iou = fp = fn = n_pred = n_gt = 0
                    for stem in val_stems:
                        entries = entries_for_config(refined_cache[stem], conf_threshold, ref_threshold, tta_mode)
                        metrics = pq_embedded(val_gt_sets[stem], entries)
                        tp += metrics["tp"]
                        fp += metrics["fp"]
                        fn += metrics["fn"]
                        sum_iou += metrics["sum_tp_iou"]
                        n_pred += len(entries)
                        n_gt += sum(len(entries_) for entries_ in val_gt_sets[stem])
                    denominator = tp + 0.5 * fp + 0.5 * fn
                    sweep_rows.append(
                        {
                            "conf": conf_threshold,
                            "ref": ref_threshold,
                            "tta": tta_mode,
                            "pq": sum_iou / denominator if denominator else 0.0,
                            "sq": sum_iou / tp if tp else 0.0,
                            "rq": tp / denominator if denominator else 0.0,
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "n_pred": n_pred,
                            "n_gt": n_gt,
                        }
                    )
        sweep_rows.sort(key=lambda row: -row["pq"])
        best = sweep_rows[0]
        log(f"sweep: best {json.dumps(best)}")
        import csv

        with (out_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_rows)
    else:
        best = {"conf": 0.20, "ref": REFINER_THRESHOLD, "tta": "guard", "pq": None}
    timings["val_inference_sweep"] = time.time() - stage_started
    stage_started = time.time()

    # ---------------- official PQ report ----------------
    pred_rows = []
    for stem in val_stems:
        entries = entries_for_config(refined_cache[stem], best["conf"], best["ref"], best["tta"])
        pred_rows.extend(entries_to_pred_rows(stem, entries))
    import pandas as pd

    pred_df = pd.DataFrame(pred_rows, columns=["filament_id", "segmentation_rle"])
    pred_df.to_csv(out_dir / "val_pred.csv", index=False)
    metric_module, metric_source = import_harness_metric(code_root)
    report = {}
    if metric_module is not None:
        gt_df = metric_module.gt_df_from_coco(data["json"], stems=val_stems)
        result = metric_module.evaluate(gt_df, pred_df)
        report = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in result.items()
            if key not in ("per_image", "tp_ious")
        }
        print(metric_module.format_report(result), flush=True)
        if "per_image" in result:
            result["per_image"].to_csv(out_dir / "val_per_image.csv", index=False)
    else:
        log("pq: WARNING harness pq_metric not importable -> embedded official formula")
        tp = sum_iou = fp = fn = 0
        n_gt_total = n_pred_total = 0
        for stem in val_stems:
            entries = entries_for_config(refined_cache[stem], best["conf"], best["ref"], best["tta"])
            metrics = pq_embedded(val_gt_sets[stem], entries)
            tp += metrics["tp"]
            fp += metrics["fp"]
            fn += metrics["fn"]
            sum_iou += metrics["sum_tp_iou"]
            n_gt_total += sum(len(item) for item in val_gt_sets[stem])
            n_pred_total += len(entries)
        denominator = tp + 0.5 * fp + 0.5 * fn
        report = {
            "pq": sum_iou / denominator if denominator else 0.0,
            "sq": sum_iou / tp if tp else 0.0,
            "rq": tp / denominator if denominator else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "sum_tp_iou": sum_iou,
            "denominator": denominator,
            "n_gt": n_gt_total,
            "n_pred": n_pred_total,
        }
    val_report = {
        "fold": args.fold,
        "fold_source": fold_source,
        "val_stems": len(val_stems),
        "best_config": best,
        "metric_source": str(metric_source),
        "pq": report.get("pq"),
        "sq": report.get("sq"),
        "rq": report.get("rq"),
        "tp": report.get("tp"),
        "fp": report.get("fp"),
        "fn": report.get("fn"),
        "n_annotator_images": report.get("n_annotator_images"),
        "n_gt": report.get("n_gt"),
        "n_pred": report.get("n_pred"),
        "timings": timings,
    }
    (out_dir / "val_report.json").write_text(json.dumps(val_report, indent=1), encoding="utf-8")
    log(f"val_report: {json.dumps(val_report)}")
    timings["val_eval"] = time.time() - stage_started
    stage_started = time.time()

    # ---------------- test inference ----------------
    test_files = sorted(data["test_images"].glob("*.jpeg")) + sorted(data["test_images"].glob("*.jpg"))
    unique_test, seen = [], set()
    for path in test_files:
        if path.name not in seen:
            seen.add(path.name)
            unique_test.append(path)
    if args.smoke:
        unique_test = unique_test[: args.test_limit or 4]
    elif args.test_limit:
        unique_test = unique_test[: args.test_limit]
    submission_rows = []
    overlap_errors = 0
    for number, image_path in enumerate(unique_test, 1):
        refined = predict_image(image_path, detector, refiner, device, best["conf"], args, imgsz=args.img_size, ref_crop=ref_crop)
        entries = entries_for_config(refined, best["conf"], best["ref"], best["tta"])
        occupied = np.zeros((H, W), dtype=bool)
        for index, entry in enumerate(entries, 1):
            full = entry_to_full(entry)
            if np.any(np.logical_and(full, occupied)):
                overlap_errors += 1
            occupied |= full
            import pycocotools.mask as mask_utils

            counts = mask_utils.encode(np.asfortranarray(full.astype(np.uint8)))["counts"]
            submission_rows.append(
                {
                    "filament_id": f"{image_path.stem}_{index}",
                    "segmentation_rle": counts.decode("ascii") if isinstance(counts, bytes) else str(counts),
                }
            )
        if number % 20 == 0:
            log(f"test inference: {number}/{len(unique_test)} images, {len(submission_rows)} rows")
    submission = pd.DataFrame(submission_rows, columns=["filament_id", "segmentation_rle"])
    if submission.empty:
        raise RuntimeError("empty submission")
    if not submission["filament_id"].is_unique:
        raise RuntimeError("duplicate filament_id in submission")
    submission.to_csv(out_dir / "submission.csv", index=False)
    n_images = int(submission["filament_id"].str.rsplit("_", n=1).str[0].nunique()) if len(submission) else 0
    log(f"submission: {len(submission)} instances / {n_images} test images, overlap_errors={overlap_errors}")
    timings["test_inference"] = time.time() - stage_started
    timings["total"] = time.time() - START_TIME

    summary = {
        "status": "ok-smoke" if args.smoke else "ok",
        "smoke": args.smoke,
        "fold": args.fold,
        "folds_source": fold_source,
        "n_train_physical": len(train_files),
        "n_val_physical": len(val_files),
        "n_train_records": n_train_records,
        "val_pq": val_report.get("pq"),
        "val_sq": val_report.get("sq"),
        "val_rq": val_report.get("rq"),
        "val_tp": val_report.get("tp"),
        "val_fp": val_report.get("fp"),
        "val_fn": val_report.get("fn"),
        "best_config": best,
        "submission_rows": len(submission),
        "submission_images": n_images,
        "overlap_errors": overlap_errors,
        "timings": {key: round(value, 1) for key, value in timings.items()},
        "detector_weights": str(detector_weights),
        "refiner_weights": str(refiner_weights),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log("=" * 100)
    log(f"SUMMARY: {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
