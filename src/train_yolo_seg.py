"""YOLO11-seg detector for the Solar Filament Segmentation 2026 challenge.

Stages (argparse sub-commands)
------------------------------
data   : build grouped 5-fold split (unit = physical jpeg), a medoid annotator
         per physical image, and ultralytics YOLO-seg datasets for each fold.
train  : train a single-class yolo11s-seg detector on one fold (or on all data
         for the submission model).
wait   : just wait for the shared GPU (work/GPU_LOCK + nvidia-smi idle).

Conventions
-----------
* Training target = the *medoid* annotator record of each physical jpeg
  (the annotator record with the highest mean mask-union IoU vs the others).
  Evaluation (PQ) always uses EVERY annotator record -> see src/infer_modeling.py.
* Images are hard-linked from data/.../train_images (same D: volume) so the
  YOLO datasets cost no extra image bytes.
* Folds: prefer harness work/cache/folds.json once available; otherwise build a
  deterministic grouped split (seed=42) and record it in
  work/cache/folds_modeling.json so it can be aligned later.
* GPU is shared with the team: always wait for work/GPU_LOCK to be absent/empty
  *and* for nvidia-smi to look idle, then write our owner tag into the lock.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMAGES = DATA / "train" / "train_images"
TEST_IMAGES = DATA / "test" / "test_images"
TRAIN_JSON = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

WORK = ROOT / "work"
CACHE = WORK / "cache"
YOLO_DATA = WORK / "yolo_data"
MODELS = WORK / "models"
GPU_LOCK = WORK / "GPU_LOCK"

IMAGE_SIZE = 2048
N_FOLDS = 5
FOLD_SEED = 42

# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)


# ----------------------------------------------------------------------------
# Shared GPU lock
# ----------------------------------------------------------------------------


def read_lock() -> str | None:
    """None = lock file absent, '' = present but empty, else its owner tag."""
    if not GPU_LOCK.exists():
        return None
    try:
        return GPU_LOCK.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def gpu_stats() -> tuple[int, int] | None:
    """(memory_used_MiB, utilization_pct) for GPU 0, or None if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=30,
        )
        line = out.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        return int(float(parts[0])), int(float(parts[1]))
    except Exception:  # noqa: BLE001 - best effort only
        return None


def gpu_idle(mem_limit: int = 2000, util_limit: int = 30, samples: int = 3) -> bool:
    for sample in range(samples):
        stats = gpu_stats()
        if stats is None:
            # No nvidia-smi: fall back to trusting the lock file.
            return True
        memory, utilization = stats
        if memory > mem_limit or utilization > util_limit:
            return False
        if sample < samples - 1:
            time.sleep(10)
    return True


def acquire_gpu_lock(owner: str = "modeling") -> bool:
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(GPU_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, owner.encode("utf-8"))
        os.close(fd)
        return True
    except FileExistsError:
        holder = read_lock()
        if holder == owner:
            return True
        if holder == "":
            # Lead may leave an empty reservation file: claim it by writing.
            GPU_LOCK.write_text(owner, encoding="utf-8")
            return True
        return False


def release_gpu_lock(owner: str = "modeling") -> None:
    if read_lock() == owner:
        try:
            GPU_LOCK.unlink()
        except OSError:
            pass


def wait_for_gpu(owner: str = "modeling", interval: int = 60, timeout_hours: float = 8.0) -> None:
    started = time.time()
    while True:
        elapsed_min = (time.time() - started) / 60.0
        if elapsed_min > timeout_hours * 60:
            raise TimeoutError(f"GPU lock not free after {timeout_hours}h")
        holder = read_lock()
        if holder == owner:
            log("GPU lock already held by this process owner, proceeding")
            return
        if holder is not None and holder != owner:
            log(f"GPU busy: lock held by '{holder}' ({elapsed_min:.0f} min waiting)")
        elif holder is None and gpu_stats() is not None and not gpu_idle(samples=1):
            mem, util = gpu_stats()
            log(f"GPU busy (no lock): {mem} MiB / {util}% util ({elapsed_min:.0f} min waiting)")
        elif gpu_idle():
            if acquire_gpu_lock(owner):
                log(f"GPU lock acquired by '{owner}' after {elapsed_min:.0f} min wait")
                return
            log("GPU idle but lock race lost, retrying")
        else:
            log(f"GPU not idle yet ({gpu_stats()}), {elapsed_min:.0f} min waiting")
        time.sleep(interval)


# ----------------------------------------------------------------------------
# COCO helpers
# ----------------------------------------------------------------------------


def load_coco() -> dict:
    log(f"loading COCO json {TRAIN_JSON} ({TRAIN_JSON.stat().st_size/1e6:.1f} MB)")
    with TRAIN_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_indices(coco: dict) -> dict:
    images_by_id: dict[str, dict] = {str(image["id"]): image for image in coco["images"]}
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
    }


def annotation_to_rle(annotation: dict, height: int, width: int) -> dict:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        return rle
    polygons = segmentation or []
    if not polygons:
        return mask_utils.encode(np.asfortranarray(np.zeros((height, width), dtype=np.uint8)))
    rles = mask_utils.frPyObjects(polygons, height, width)
    return mask_utils.merge(rles)


def annotation_to_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    return mask_utils.decode(annotation_to_rle(annotation, height, width)).astype(bool)


def union_rle_for_record(image_id: str, indices: dict) -> dict:
    info = indices["images_by_id"][image_id]
    height, width = int(info["height"]), int(info["width"])
    rles = [
        annotation_to_rle(annotation, height, width)
        for annotation in indices["annotations_by_image"].get(image_id, [])
        if not annotation.get("iscrowd", 0)
    ]
    if not rles:
        return mask_utils.encode(np.asfortranarray(np.zeros((height, width), dtype=np.uint8)))
    return mask_utils.merge(rles)


def choose_medoid_annotators(indices: dict) -> dict[str, str]:
    """filename -> medoid image_id (the annotator record closest to the others)."""
    cache_path = YOLO_DATA / "meta" / "medoid.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        log(f"medoid: loaded cache ({len(payload)} files)")
        return {str(k): str(v) for k, v in payload.items()}

    selected: dict[str, str] = {}
    multi = 0
    for filename, image_ids in sorted(indices["image_ids_by_file"].items()):
        if len(image_ids) == 1:
            selected[filename] = image_ids[0]
            continue
        multi += 1
        unions = [union_rle_for_record(image_id, indices) for image_id in image_ids]
        pairwise = np.asarray(mask_utils.iou(unions, unions, [0] * len(unions)), dtype=np.float64)
        mean_agreement = (pairwise.sum(axis=1) - 1.0) / max(len(image_ids) - 1, 1)
        selected[filename] = image_ids[int(np.argmax(mean_agreement))]
    write_json(cache_path, selected)
    log(f"medoid: {len(selected)} physical images ({multi} multi-annotator) -> {cache_path}")
    return selected


# ----------------------------------------------------------------------------
# Grouped folds
# ----------------------------------------------------------------------------


def _fold_of_from_payload(payload) -> dict[str, int] | None:
    if not isinstance(payload, dict):
        return None
    if "fold_of" in payload and isinstance(payload["fold_of"], dict):
        return {str(k): int(v) for k, v in payload["fold_of"].items()}
    for key in ("folds", "assignments", "observations", "records"):
        value = payload.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            result = {}
            for row in value:
                filename = row.get("filename") or row.get("file_name") or row.get("stem")
                fold = row.get("fold")
                if filename is None or fold is None:
                    return None
                result[str(filename)] = int(fold)
            return result
        if isinstance(value, dict) and value:
            result = {}
            ok = True
            for fold, files in value.items():
                if not isinstance(files, list):
                    ok = False
                    break
                for filename in files:
                    result[str(filename)] = int(fold)
            if ok and result:
                return result
    if payload and all(isinstance(v, (int, np.integer)) for v in payload.values()):
        return {str(k): int(v) for k, v in payload.items()}
    return None


def _align_folds(folds: dict[str, int] | None, filenames: list[str]) -> dict[str, int] | None:
    """Map a fold table keyed by stem OR by file_name onto the given file names."""
    if not folds:
        return None
    aligned: dict[str, int] = {}
    for filename in filenames:
        stem = Path(filename).stem
        if filename in folds:
            aligned[filename] = int(folds[filename])
        elif stem in folds:
            aligned[filename] = int(folds[stem])
        else:
            return None
    return aligned


def load_grouped_folds(filenames: list[str]) -> tuple[dict[str, int], str]:
    """Return (filename -> fold, source description). Harness folds.json wins."""
    harness_path = CACHE / "folds.json"
    if harness_path.exists():
        try:
            payload = json.loads(harness_path.read_text(encoding="utf-8"))
            folds = _align_folds(_fold_of_from_payload(payload), filenames)
            if folds:
                log(f"folds: using harness {harness_path} ({len(folds)} files)")
                return folds, f"harness:{harness_path}"
            log("folds: harness file present but not directly usable, falling back")
        except Exception as error:  # noqa: BLE001
            log(f"folds: failed to parse harness file ({error}), falling back")

    modeling_path = CACHE / "folds_modeling.json"
    if modeling_path.exists():
        payload = json.loads(modeling_path.read_text(encoding="utf-8"))
        folds = _align_folds(_fold_of_from_payload(payload), filenames)
        if folds:
            log(f"folds: using cached {modeling_path}")
            return folds, f"modeling:{modeling_path}"

    files = np.array(sorted(filenames))
    rng = np.random.default_rng(FOLD_SEED)
    permutation = rng.permutation(len(files))
    fold_assignment = np.empty(len(files), dtype=int)
    fold_assignment[permutation] = np.arange(len(files)) % N_FOLDS
    folds = {str(f): int(k) for f, k in zip(files, fold_assignment)}
    write_json(
        modeling_path,
        {
            "seed": FOLD_SEED,
            "n_folds": N_FOLDS,
            "unit": "physical jpeg (grouped, no annotator leakage)",
            "source": "modeling fallback (harness work/cache/folds.json not available)",
            "fold_of": folds,
        },
    )
    log(f"folds: built grouped split seed={FOLD_SEED} -> {modeling_path}")
    return folds, f"modeling:{modeling_path}"


# ----------------------------------------------------------------------------
# YOLO dataset writers
# ----------------------------------------------------------------------------


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
    # YOLO-seg expects one polygon per line: keep the largest part.
    polygon = max(polygons, key=polygon_area)
    normalized = np.clip(polygon / float(IMAGE_SIZE), 0.0, 1.0)
    return "0 " + " ".join(f"{value:.6f}" for value in normalized.ravel())


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_yolo_split(
    split_dir: Path,
    filenames: list[str],
    medoid: dict[str, str],
    indices: dict,
    train_image_dir: Path,
) -> tuple[int, int]:
    image_output = split_dir / "images"
    label_output = split_dir / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for filename in filenames:
        source = train_image_dir / filename
        if not source.exists():
            raise FileNotFoundError(source)
        link_or_copy(source, image_output / filename)
        image_id = medoid[filename]
        lines = []
        for annotation in indices["annotations_by_image"].get(image_id, []):
            if annotation.get("iscrowd", 0):
                continue
            line = annotation_to_yolo_line(annotation)
            if line is not None:
                lines.append(line)
        n_boxes += len(lines)
        (label_output / f"{Path(filename).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    return len(filenames), n_boxes


def prepare_fold_data(
    fold: int,
    folds: dict[str, int],
    medoid: dict[str, str],
    indices: dict,
    force: bool = False,
) -> Path:
    """fold == -1 -> train+val on all files (submission model; val is fold0)."""
    name = f"fold_{fold}" if fold >= 0 else "full"
    fold_dir = YOLO_DATA / name
    yaml_path = fold_dir / "data.yaml"
    marker = fold_dir / ".ready"
    if marker.exists() and not force:
        log(f"yolo data {name}: already prepared -> {yaml_path}")
        return yaml_path

    filenames = sorted(folds)
    if fold >= 0:
        train_files = [f for f in filenames if folds[f] != fold]
        val_files = [f for f in filenames if folds[f] == fold]
    else:
        train_files = filenames
        val_files = [f for f in filenames if folds[f] == 0]

    started = time.time()
    n_train, b_train = write_yolo_split(fold_dir / "train", train_files, medoid, indices, TRAIN_IMAGES)
    n_val, b_val = write_yolo_split(fold_dir / "val", val_files, medoid, indices, TRAIN_IMAGES)
    yaml_path.write_text(
        f"path: {fold_dir.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n  0: filament\n",
        encoding="utf-8",
    )
    marker.write_text("ready", encoding="utf-8")
    log(
        f"yolo data {name}: train {n_train} imgs / {b_train} boxes, "
        f"val {n_val} imgs / {b_val} boxes in {time.time()-started:.1f}s -> {fold_dir}"
    )
    return yaml_path


# ----------------------------------------------------------------------------
# All-annotations dataset (learn the annotator consensus geometry)
# ----------------------------------------------------------------------------


def sample_stem_for(image_id: str, filename: str) -> str:
    annotator = image_id.split("-", 1)[0]
    return f"{Path(filename).stem}__{annotator}"


def build_all_images_cache(indices: dict, resize: int, force: bool = False) -> dict[str, dict]:
    """One resized (resize x resize) JPEG per annotation record, shared by every fold.

    A physical image with 2-3 annotator records becomes 2-3 training samples with
    different masks: the model then learns the annotator consensus instead of one
    arbitrary annotator's geometry (Lead's strategic update, FACTS.md section 10).
    """
    cache_dir = YOLO_DATA / f"images_all_{resize}"
    meta_path = cache_dir / "meta.json"
    if meta_path.exists() and not force:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        log(f"all-annotation image cache: loaded {len(payload)} samples from {meta_path}")
        return payload
    cache_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, dict] = {}
    started = time.time()
    files = sorted(indices["image_ids_by_file"])
    for file_index, filename in enumerate(files, 1):
        image = cv2.imread(str(TRAIN_IMAGES / filename), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(filename)
        if image.shape[0] != resize or image.shape[1] != resize:
            image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_AREA)
        for image_id in indices["image_ids_by_file"][filename]:
            sample_stem = sample_stem_for(image_id, filename)
            destination = cache_dir / f"{sample_stem}.jpg"
            if force or not destination.exists():
                cv2.imwrite(str(destination), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            mapping[sample_stem] = {"filename": filename, "image_id": image_id, "path": str(destination)}
        if file_index % 200 == 0:
            log(f"all-annotation image cache: {file_index}/{len(files)} physical images")
    write_json(meta_path, mapping)
    log(
        f"all-annotation image cache: {len(mapping)} samples at {resize}px in "
        f"{time.time()-started:.1f}s -> {cache_dir}"
    )
    return mapping


def write_all_split(
    split_dir: Path, filenames: list[str], all_images: dict[str, dict], indices: dict
) -> tuple[int, int]:
    file_set = set(filenames)
    image_output = split_dir / "images"
    label_output = split_dir / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    n_images = 0
    n_boxes = 0
    for sample_stem, sample in all_images.items():
        if sample["filename"] not in file_set:
            continue
        link_or_copy(Path(sample["path"]), image_output / f"{sample_stem}.jpg")
        lines = []
        for annotation in indices["annotations_by_image"].get(sample["image_id"], []):
            if annotation.get("iscrowd", 0):
                continue
            line = annotation_to_yolo_line(annotation)
            if line is not None:
                lines.append(line)
        n_boxes += len(lines)
        (label_output / f"{sample_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_images += 1
    return n_images, n_boxes


def prepare_all_data(
    fold: int,
    folds: dict[str, int],
    indices: dict,
    resize: int,
    all_images: dict | None = None,
    force: bool = False,
) -> Path:
    """YOLO dataset where each sample is one (physical image, annotation record) pair."""
    name = f"all_{fold}" if fold >= 0 else "all_full"
    fold_dir = YOLO_DATA / name
    yaml_path = fold_dir / "data.yaml"
    marker = fold_dir / ".ready"
    if marker.exists() and not force:
        log(f"yolo data {name}: already prepared -> {yaml_path}")
        return yaml_path
    if all_images is None:
        all_images = build_all_images_cache(indices, resize)
    filenames = sorted(folds)
    if fold >= 0:
        train_files = [f for f in filenames if folds[f] != fold]
        val_files = [f for f in filenames if folds[f] == fold]
    else:
        train_files = filenames
        val_files = [f for f in filenames if folds[f] == 0]
    started = time.time()
    n_train, b_train = write_all_split(fold_dir / "train", train_files, all_images, indices)
    n_val, b_val = write_all_split(fold_dir / "val", val_files, all_images, indices)
    yaml_path.write_text(
        f"path: {fold_dir.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n  0: filament\n",
        encoding="utf-8",
    )
    marker.write_text("ready", encoding="utf-8")
    log(
        f"yolo data {name}: train {n_train} samples / {b_train} boxes, "
        f"val {n_val} samples / {b_val} boxes in {time.time()-started:.1f}s -> {fold_dir}"
    )
    return yaml_path


# ----------------------------------------------------------------------------
# Detector training
# ----------------------------------------------------------------------------


def train_detector(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    coco = load_coco()
    indices = build_indices(coco)
    filenames = sorted(indices["image_ids_by_file"])
    log(f"physical images: {len(filenames)}; annotator-image records: {len(indices['images_by_id'])}")
    medoid = choose_medoid_annotators(indices)
    folds, fold_source = load_grouped_folds(filenames)
    label_source = getattr(args, "label_source", "all")
    resize = getattr(args, "resize", 1024)
    if label_source == "all":
        yaml_path = prepare_all_data(args.fold, folds, indices, resize, force=args.force_data)
    else:
        yaml_path = prepare_fold_data(args.fold, folds, medoid, indices, force=args.force_data)

    suffix = "" if label_source == "medoid" else f"_{label_source}"
    output_dir = MODELS / "yolo" / (("full" if args.fold < 0 else f"fold_{args.fold}") + suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    if best_path.exists() and not args.force_train:
        log(f"detector already trained: {best_path} (use --force-train to redo)")
        return best_path

    if args.wait_gpu:
        wait_for_gpu(owner="modeling")
    else:
        holder = read_lock()
        if holder not in (None, "", "modeling"):
            raise RuntimeError(f"GPU_LOCK is held by '{holder}'; pass --wait-gpu to wait")
        if holder is None:
            acquire_gpu_lock("modeling")

    run_name = f"yolo11s_seg_{label_source}_{'full' if args.fold < 0 else 'fold_' + str(args.fold)}"
    model = YOLO(args.model)
    started = time.time()
    elapsed = 0.0
    try:
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=0 if args.device == "0" else args.device,
            workers=args.workers,
            seed=args.seed + max(args.fold, 0),
            patience=args.patience,
            project=str(MODELS / "yolo" / "runs"),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            single_cls=True,
            mask_ratio=args.mask_ratio,
            overlap_mask=False,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.25,
            fliplr=0.5,
            flipud=0.5,
            degrees=args.degrees,
            scale=args.scale,
            translate=args.translate,
            mosaic=args.mosaic,
            close_mosaic=args.close_mosaic,
            cache=args.cache,
            amp=True,
            plots=False,
            val=not getattr(args, "no_val", False),
            fraction=args.fraction,
            verbose=True,
        )
    finally:
        elapsed = time.time() - started
        log(f"detector training finished in {elapsed/60:.1f} min")
        release_gpu_lock("modeling")

    trainer = getattr(model, "trainer", None)
    source_best = Path(trainer.best) if trainer is not None and getattr(trainer, "best", None) else None
    weights_dir = MODELS / "yolo" / "runs" / run_name / "weights"
    if source_best is None or not source_best.exists():
        for candidate in (weights_dir / "best.pt", weights_dir / "last.pt"):
            if candidate.exists():
                source_best = candidate
                break
    if source_best is None or not source_best.exists():
        raise FileNotFoundError(f"trained weights not found under {weights_dir}")
    shutil.copy2(source_best, best_path)
    summary = {
        "fold": args.fold,
        "label_source": label_source,
        "resize": resize,
        "model": args.model,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "epochs": args.epochs,
        "mask_ratio": args.mask_ratio,
        "fraction": args.fraction,
        "fold_source": fold_source,
        "train_seconds": round(elapsed, 1),
        "best_pt": str(best_path),
        "run_dir": str(MODELS / "yolo" / "runs" / run_name),
    }
    write_json(output_dir / "train_summary.json", summary)
    log(f"detector saved: {best_path}")
    log(json.dumps(summary, indent=1))
    return best_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="prepare grouped folds + YOLO-seg datasets")
    data.add_argument("--folds", default="all", help="'all' or comma separated fold ids")
    data.add_argument("--label-source", default="all", choices=["all", "medoid"])
    data.add_argument("--resize", type=int, default=1024, help="cache resize for all-annotations dataset")
    data.add_argument("--force", action="store_true")

    train = sub.add_parser("train", help="train the YOLO11-seg detector")
    train.add_argument("--fold", type=int, default=0, help="-1 = all data (submission model)")
    train.add_argument("--model", default="yolo11s-seg.pt")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--imgsz", type=int, default=1024)
    train.add_argument("--batch", type=int, default=4)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--mask-ratio", type=int, default=2)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--cache", default="None", help="None / ram / disk")
    train.add_argument("--fraction", type=float, default=1.0, help="subset fraction (smoke tests)")
    train.add_argument("--device", default="0")
    train.add_argument("--label-source", default="all", choices=["all", "medoid"])
    train.add_argument("--resize", type=int, default=1024)
    train.add_argument("--degrees", type=float, default=0.0)
    train.add_argument("--scale", type=float, default=0.20)
    train.add_argument("--translate", type=float, default=0.05)
    train.add_argument("--mosaic", type=float, default=0.0)
    train.add_argument("--close-mosaic", type=int, default=5)
    train.add_argument("--no-val", action="store_true", help="skip per-epoch val (faster; use last.pt)")
    train.add_argument("--wait-gpu", action="store_true")
    train.add_argument("--force-data", action="store_true")
    train.add_argument("--force-train", action="store_true")

    wait = sub.add_parser("wait", help="wait for the shared GPU")
    wait.add_argument("--interval", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "wait":
        wait_for_gpu(owner="modeling", interval=args.interval)
        return 0

    if args.command == "data":
        coco = load_coco()
        indices = build_indices(coco)
        filenames = sorted(indices["image_ids_by_file"])
        medoid = choose_medoid_annotators(indices)
        folds, fold_source = load_grouped_folds(filenames)
        write_json(
            YOLO_DATA / "meta" / "fold_summary.json",
            {
                "source": fold_source,
                "counts": {str(k): sum(1 for v in folds.values() if v == k) for k in range(N_FOLDS)},
            },
        )
        if args.folds == "all":
            fold_ids = list(range(N_FOLDS)) + [-1]
        else:
            fold_ids = [int(value) for value in args.folds.split(",")]
        label_source = getattr(args, "label_source", "all")
        resize = getattr(args, "resize", 1024)
        all_images = build_all_images_cache(indices, resize) if label_source == "all" else None
        for fold in fold_ids:
            if label_source == "all":
                prepare_all_data(fold, folds, indices, resize, all_images, force=args.force)
            else:
                prepare_fold_data(fold, folds, medoid, indices, force=args.force)
        return 0

    if args.command == "train":
        if args.cache in ("None", "none", ""):
            args.cache = False
        train_detector(args)
        return 0

    raise ValueError(args.command)


if __name__ == "__main__":
    sys.exit(main())