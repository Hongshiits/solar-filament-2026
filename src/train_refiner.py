"""Seed-conditioned crop refiner for Solar Filament Segmentation 2026.

Recipe (aligned with the public 0.70 handoff + the public YOLO11-seg PQ notebook)
-------------------------------------------------------------------------------
* The YOLO11-seg detector gives coarse instance masks at 1024 input resolution.
* For every detected instance we cut a square crop from the FULL-RES 2048 image
  (context = 1.8 x max side of the seed mask, min 96 px), resize it to 256,
  and feed a 3-channel tensor [grayscale, CLAHE, seed mask] to a ResNet18 U-Net.
* The network predicts one 1-channel probability map -> refined mask at 2048.
* Training crops are pre-computed once per fold into uint8 memmaps
  (work/yolo_data/refiner_cache/...) so an epoch only costs a few seconds.
  Seed masks are synthesised at training time by jittering the GT mask
  (erode/dilate, shift, coarse downsample-upsample to mimic YOLO masks) or by
  using a filled padded box, so the refiner is robust to detector error.

Usage
-----
python src/train_refiner.py cache --fold 0
python src/train_refiner.py train --fold 0 --epochs 20 --wait-gpu
python src/train_refiner.py cache --fold -1   # all-data submission model
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_yolo_seg import (  # noqa: E402
    GPU_LOCK,
    MODELS,
    YOLO_DATA,
    acquire_gpu_lock,
    annotation_to_mask,
    build_indices,
    choose_medoid_annotators,
    load_coco,
    load_grouped_folds,
    log,
    read_lock,
    release_gpu_lock,
    wait_for_gpu,
    write_json,
)

REFINER_CROP = 256
CONTEXT = 1.8
MIN_CROP = 96
REFINER_THRESHOLD = 0.65
CACHE_ROOT = YOLO_DATA / "refiner_cache"


# ----------------------------------------------------------------------------
# Geometry / seed synthesis
# ----------------------------------------------------------------------------


def square_bounds_from_mask(mask: np.ndarray, context: float = CONTEXT, minimum: int = MIN_CROP):
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


def _erode_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return mask
    kernel_size = 2 * abs(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if radius > 0:
        return cv2.dilate(mask, kernel)
    return cv2.erode(mask, kernel)


def _shift(mask: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)


def coarsen_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Downsample+upsample like an upscaled low-res YOLO mask (nearest, keeps hard edges)."""
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
    """Training-time fake detector seed. target: uint8 {0,1} crop."""
    mode = rng.choice(["mask", "mask", "mask", "coarse", "box"])
    target_bool = target > 0
    if mode == "box":
        return box_seed(target_bool, float(rng.uniform(0.15, 0.6))).astype(np.uint8)
    seed = target_bool.astype(np.uint8)
    radius = int(rng.integers(-6, 7))
    seed = _erode_dilate(seed, radius)
    if mode == "coarse":
        seed = coarsen_mask(seed, int(rng.integers(2, 5)))
    if rng.random() < 0.7:
        limit = max(4, int(0.12 * target.shape[0]))
        seed = _shift(seed, int(rng.integers(-limit, limit + 1)), int(rng.integers(-limit, limit + 1)))
    if seed.sum() == 0:
        seed = target_bool.astype(np.uint8)
    return seed


# ----------------------------------------------------------------------------
# Crop cache (uint8 memmaps)
# ----------------------------------------------------------------------------


def cache_dir(fold: int, crop: int = REFINER_CROP, label_source: str = "all") -> Path:
    return CACHE_ROOT / (f"fold_{fold}" if fold >= 0 else "full") / f"{label_source}_c{crop}"


def _cache_paths(
    fold: int, split: str, crop: int = REFINER_CROP, label_source: str = "all"
) -> tuple[Path, Path, Path]:
    directory = cache_dir(fold, crop, label_source)
    return directory / f"{split}_images.npy", directory / f"{split}_targets.npy", directory / f"{split}_meta.json"


def record_ids_for_file(filename: str, indices: dict, label_source: str, medoid: dict[str, str]) -> list[str]:
    """Annotation records (image_ids) used as training targets for one physical image."""
    if label_source == "medoid":
        return [medoid[filename]]
    return list(indices["image_ids_by_file"][filename])


def _count_annotations(
    files: list[str], medoid: dict[str, str], indices: dict, label_source: str = "all"
) -> int:
    total = 0
    for filename in files:
        for image_id in record_ids_for_file(filename, indices, label_source, medoid):
            for annotation in indices["annotations_by_image"].get(image_id, []):
                if annotation.get("iscrowd", 0):
                    continue
                if not annotation.get("segmentation"):
                    continue
                total += 1
    return total


def build_split_cache(
    fold: int,
    split: str,
    files: list[str],
    medoid: dict[str, str],
    indices: dict,
    crop: int = REFINER_CROP,
    force: bool = False,
    label_source: str = "all",
) -> tuple[Path, Path]:
    images_path, targets_path, meta_path = _cache_paths(fold, split, crop, label_source)
    if meta_path.exists() and images_path.exists() and targets_path.exists() and not force:
        log(f"refiner cache {fold}/{split}/{label_source}: already exists ({meta_path.parent})")
        return images_path, targets_path

    started = time.time()
    total = _count_annotations(files, medoid, indices, label_source)
    images_path.parent.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(images_path, mode="w+", dtype=np.uint8, shape=(total, crop, crop))
    targets = np.lib.format.open_memmap(targets_path, mode="w+", dtype=np.uint8, shape=(total, crop, crop))
    meta = {
        "fold": fold,
        "split": split,
        "crop": crop,
        "label_source": label_source,
        "files": files,
        "samples": [],
    }
    cursor = 0
    for file_index, filename in enumerate(files, 1):
        image_path = Path(indices["image_paths"][filename])
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape
        for image_id in record_ids_for_file(filename, indices, label_source, medoid):
            for annotation in indices["annotations_by_image"].get(image_id, []):
                if annotation.get("iscrowd", 0):
                    continue
                if not annotation.get("segmentation"):
                    continue
                mask = annotation_to_mask(annotation, height, width)
                if not mask.any():
                    continue
                x0, y0, x1, y1 = square_bounds_from_mask(mask, CONTEXT, MIN_CROP)
                image_crop = cv2.resize(image[y0:y1, x0:x1], (crop, crop), interpolation=cv2.INTER_LINEAR)
                target_crop = cv2.resize(
                    mask[y0:y1, x0:x1].astype(np.uint8), (crop, crop), interpolation=cv2.INTER_NEAREST
                )
                images[cursor] = image_crop
                targets[cursor] = target_crop
                meta["samples"].append(
                    {
                        "filename": filename,
                        "image_id": image_id,
                        "annotation_id": str(annotation["id"]),
                        "cursor": cursor,
                    }
                )
                cursor += 1
        if file_index % 100 == 0:
            log(f"refiner cache {fold}/{split}/{label_source}: {file_index}/{len(files)} files, {cursor} samples")
    if cursor != total:
        images = images[:cursor]
        targets = targets[:cursor]
        meta["n_samples"] = cursor
    images.flush()
    targets.flush()
    del images, targets
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    log(
        f"refiner cache {fold}/{split}/{label_source}: {cursor} samples in "
        f"{time.time()-started:.1f}s -> {images_path.parent}"
    )
    return images_path, targets_path


def load_indices_with_paths() -> tuple[dict, dict[str, str], list[str]]:
    from train_yolo_seg import TRAIN_IMAGES

    coco = load_coco()
    indices = build_indices(coco)
    indices["image_paths"] = {
        filename: str(TRAIN_IMAGES / filename) for filename in indices["image_ids_by_file"]
    }
    filenames = sorted(indices["image_ids_by_file"])
    medoid = choose_medoid_annotators(indices)
    return indices, medoid, filenames


# ----------------------------------------------------------------------------
# Dataset / model
# ----------------------------------------------------------------------------


class CachedRefinerDataset(Dataset):
    def __init__(self, images_path: Path, targets_path: Path, augment: bool, seed: int = 42):
        self.images = np.load(images_path, mmap_mode="r")
        self.targets = np.load(targets_path, mmap_mode="r")
        self.augment = augment
        self.seed = seed
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
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
            # crop-context zoom jitter: detector seeds have a different size than the
            # GT mask used to build the cache -> emulate zoom in/out by +-15%.
            zoom = random.uniform(0.85, 1.15)
            if abs(zoom - 1.0) > 0.02:
                side = max(8, int(round(REFINER_CROP * zoom)))
                image_resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_LINEAR)
                clahe_resized = cv2.resize(clahe, (side, side), interpolation=cv2.INTER_LINEAR)
                seed_resized = cv2.resize(seed_crop, (side, side), interpolation=cv2.INTER_NEAREST)
                target_resized = cv2.resize(target, (side, side), interpolation=cv2.INTER_NEAREST)
                if side >= REFINER_CROP:
                    offset = (side - REFINER_CROP) // 2
                    image_resized = image_resized[offset : offset + REFINER_CROP, offset : offset + REFINER_CROP]
                    clahe_resized = clahe_resized[offset : offset + REFINER_CROP, offset : offset + REFINER_CROP]
                    seed_resized = seed_resized[offset : offset + REFINER_CROP, offset : offset + REFINER_CROP]
                    target_resized = target_resized[offset : offset + REFINER_CROP, offset : offset + REFINER_CROP]
                else:
                    pad = REFINER_CROP - side
                    top, left = pad // 2, pad // 2
                    image_resized = np.pad(image_resized, ((top, pad - top), (left, pad - left)), mode="reflect")
                    clahe_resized = np.pad(clahe_resized, ((top, pad - top), (left, pad - left)), mode="reflect")
                    seed_resized = np.pad(seed_resized, ((top, pad - top), (left, pad - left)), mode="constant")
                    target_resized = np.pad(target_resized, ((top, pad - top), (left, pad - left)), mode="constant")
                image, clahe, seed_crop, target = image_resized, clahe_resized, seed_resized, target_resized
                features = np.stack([image, clahe, seed_crop * 255], axis=0).astype(np.float32) / 255.0
            k = random.randrange(4)
            features = np.rot90(features, k, axes=(1, 2)).copy()
            target = np.rot90(target, k).copy()
            if random.random() < 0.5:
                features = features[:, :, ::-1].copy()
                target = target[:, ::-1].copy()
            gain = random.uniform(0.9, 1.1)
            features[:2] = np.clip(features[:2] * gain, 0.0, 1.0)
        return torch.from_numpy(features), torch.from_numpy(target[None].astype(np.float32))


class ConvBlock(nn.Sequential):
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
    """ResNet18 U-Net (matches the public 0.70 CropUNet), 3-channel input."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = ConvBlock(96, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
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


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def batch_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    predictions = logits.sigmoid() >= threshold
    targets_bool = targets > 0.5
    intersection = (predictions & targets_bool).sum(dim=(1, 2, 3)).float()
    denominator = predictions.sum(dim=(1, 2, 3)) + targets_bool.sum(dim=(1, 2, 3))
    return float(((2 * intersection + 1e-7) / (denominator + 1e-7)).mean().item())


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------


def train_refiner(args: argparse.Namespace) -> Path:
    indices, medoid, filenames = load_indices_with_paths()
    folds, fold_source = load_grouped_folds(filenames)
    fold = args.fold
    if fold >= 0:
        train_files = [f for f in filenames if folds[f] != fold]
        val_files = [f for f in filenames if folds[f] == fold]
    else:
        train_files = filenames
        val_files = [f for f in filenames if folds[f] == 0]

    label_source = getattr(args, "label_source", "all")
    train_images, train_targets = build_split_cache(
        fold, "train", train_files, medoid, indices, args.crop, label_source=label_source
    )
    val_images, val_targets = build_split_cache(
        fold, "val", val_files, medoid, indices, args.crop, label_source=label_source
    )

    suffix = "" if label_source == "medoid" else f"_{label_source}"
    output_dir = MODELS / "refiner" / (("full" if fold < 0 else f"fold_{fold}") + suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.pt"
    if checkpoint.exists() and not args.force_train:
        log(f"refiner already trained: {checkpoint} (use --force-train to redo)")
        return checkpoint

    if args.wait_gpu:
        wait_for_gpu(owner="modeling")
    else:
        holder = read_lock()
        if holder not in (None, "", "modeling"):
            raise RuntimeError(f"GPU_LOCK is held by '{holder}'; pass --wait-gpu")
        if holder is None:
            acquire_gpu_lock("modeling")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = CachedRefinerDataset(train_images, train_targets, augment=True, seed=args.seed)
    val_dataset = CachedRefinerDataset(val_images, val_targets, augment=False, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch * 2, shuffle=False, num_workers=0)

    pretrained = True
    if getattr(args, "ref_pretrained", "auto") == "false":
        pretrained = False
    elif getattr(args, "ref_pretrained", "auto") == "auto":
        checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
        pretrained = checkpoint_path.exists()
        if not pretrained:
            log("refiner: resnet18 ImageNet weights not cached -> training from scratch (random init)")
    else:
        pretrained = bool(getattr(args, "ref_pretrained", "true") == "true")
    model = SeedRefiner(pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_dice = -1.0
    history = []
    started = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses, train_dices = [], []
            epoch_started = time.time()
            for features, targets in train_loader:
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(features)
                    loss = 0.45 * F.binary_cross_entropy_with_logits(logits, targets) + 0.55 * soft_dice_loss(
                        logits, targets
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))
                train_dices.append(batch_dice(logits.detach(), targets))
            model.eval()
            val_losses, val_dices = [], []
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(device), targets.to(device)
                    logits = model(features)
                    loss = 0.45 * F.binary_cross_entropy_with_logits(logits, targets) + 0.55 * soft_dice_loss(
                        logits, targets
                    )
                    val_losses.append(float(loss.item()))
                    val_dices.append(batch_dice(logits, targets))
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "train_dice": float(np.mean(train_dices)),
                "val_loss": float(np.mean(val_losses)),
                "val_dice": float(np.mean(val_dices)),
                "seconds": round(time.time() - epoch_started, 1),
            }
            history.append(row)
            log(
                f"refiner fold={fold} epoch {epoch:02d}/{args.epochs} "
                f"train loss={row['train_loss']:.4f} dice={row['train_dice']:.4f} | "
                f"val loss={row['val_loss']:.4f} dice={row['val_dice']:.4f} ({row['seconds']}s)"
            )
            if row["val_dice"] > best_dice:
                best_dice = row["val_dice"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_dice": best_dice,
                        "config": {
                            "crop": args.crop,
                            "context": CONTEXT,
                            "min_crop": MIN_CROP,
                            "augment": "jittered/coarsened mask + box seeds",
                        },
                    },
                    checkpoint,
                )
                log(f"  saved {checkpoint} (val_dice={best_dice:.4f})")
    finally:
        release_gpu_lock("modeling")
        elapsed = time.time() - started
        log(f"refiner training finished in {elapsed/60:.1f} min, best val dice={best_dice:.4f}")

    write_json(
        output_dir / "train_summary.json",
        {
            "fold": fold,
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "crop": args.crop,
            "train_samples": int(len(train_dataset)),
            "val_samples": int(len(val_dataset)),
            "best_val_dice": best_dice,
            "train_seconds": round(elapsed, 1),
            "fold_source": fold_source,
            "history": history,
        },
    )
    return checkpoint


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("cache", help="precompute crop caches for a fold")
    cache.add_argument("--fold", type=int, default=0)
    cache.add_argument("--crop", type=int, default=REFINER_CROP)
    cache.add_argument("--label-source", default="all", choices=["all", "medoid"])
    cache.add_argument("--force", action="store_true")

    train = sub.add_parser("train", help="train the crop refiner")
    train.add_argument("--fold", type=int, default=0, help="-1 = all data")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch", type=int, default=24)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--crop", type=int, default=REFINER_CROP)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--label-source", default="all", choices=["all", "medoid"])
    train.add_argument("--ref-pretrained", default="auto", choices=["auto", "true", "false"])
    train.add_argument("--wait-gpu", action="store_true")
    train.add_argument("--force-train", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "cache":
        indices, medoid, filenames = load_indices_with_paths()
        folds, fold_source = load_grouped_folds(filenames)
        fold = args.fold
        if fold >= 0:
            train_files = [f for f in filenames if folds[f] != fold]
            val_files = [f for f in filenames if folds[f] == fold]
        else:
            train_files = filenames
            val_files = [f for f in filenames if folds[f] == 0]
        label_source = getattr(args, "label_source", "all")
        build_split_cache(
            fold, "train", train_files, medoid, indices, args.crop, force=args.force, label_source=label_source
        )
        build_split_cache(
            fold, "val", val_files, medoid, indices, args.crop, force=args.force, label_source=label_source
        )
        log(f"refiner cache ready (fold={fold}, source={fold_source}, labels={label_source})")
        return 0
    if args.command == "train":
        train_refiner(args)
        return 0
    raise ValueError(args.command)


if __name__ == "__main__":
    sys.exit(main())