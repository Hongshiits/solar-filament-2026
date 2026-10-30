"""Kaggle-kernel entrypoint: semantic U-Net cross-fold probabilities.

Two products (selectable):
  1) on-fold-N val probabilities for every provided checkpoint
     (e.g. fold1/fold2/fold3/fold4 models on fold0 val images) -> optimistic
     multi-fold-average proxy evaluation by harness;
  2) optionally, TEST-set probabilities for every provided checkpoint
     (multi-fold average source for dariush x sem fusion).

Self-contained: only needs torch/torchvision/smp/numpy/cv2/pandas on the Kaggle
image (smp = segmentation-models-pytorch) and the competition data mounted.

    # cross-fold probs on fold0 val
    python cloud_sem_crossfold_probs.py --ckpt a.pt b.pt --target-fold 0         --out-dir /kaggle/working/sem_crossfold
    # test probs
    python cloud_sem_crossfold_probs.py --ckpt a.pt b.pt --test         --out-dir /kaggle/working/sem_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

H = W = 2048
DISK_R = 950
YY, XX = np.ogrid[:H, :W]
DISK = ((YY - (H - 1) / 2.0) ** 2 + (XX - (W - 1) / 2.0) ** 2) <= DISK_R ** 2
INPUT = Path("/kaggle/input")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)




def ensure_packages() -> None:
    # Install optional packages that are missing from the Kaggle image (internet is enabled).
    import importlib
    import subprocess

    needed = []
    for module, package in (
        ("segmentation_models_pytorch", "segmentation-models-pytorch"),
        ("timm", "timm"),
    ):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            needed.append(package)
    if needed:
        print("[bootstrap] pip install " + " ".join(needed), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-input", *needed])


def disk_stats(image: np.ndarray):
    values = image[DISK].astype(np.float32) / 255.0
    return float(values.mean()), max(float(values.std()), 1e-3)


def find_data_root(explicit: str | None = None) -> dict:
    if explicit:
        root = Path(explicit)
    else:
        candidates = []
        if INPUT.exists():
            candidates += sorted(INPUT.glob("**/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
        if not candidates:
            raise FileNotFoundError("competition data not found under /kaggle/input")
        root = candidates[0].parent.parent
    return {
        "root": root,
        "train_images": root / "train" / "train_images",
        "test_images": root / "test" / "test_images",
        "json": root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    }


def find_folds(explicit: str | None = None) -> dict:
    if explicit:
        return json.load(open(explicit, encoding="utf-8"))
    candidates = []
    if INPUT.exists():
        candidates += sorted(INPUT.glob("**/work/cache/folds.json")) + sorted(INPUT.glob("**/folds.json"))
    if not candidates:
        raise FileNotFoundError("folds.json not found under /kaggle/input")
    return json.load(open(candidates[0], encoding="utf-8"))


def load_model(ckpt_path: Path, device: torch.device):
    import segmentation_models_pytorch as smp

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    encoder = payload.get("encoder", "tu-tf_efficientnetv2_s") if isinstance(payload, dict) else "tu-tf_efficientnetv2_s"
    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(state)
    model = model.to(device).eval()
    log(f"loaded {ckpt_path.name} encoder={encoder}")
    return model


@torch.no_grad()
def full_probability_tiled(model, image: np.ndarray, device: torch.device, tile: int, overlap: int, tta: int) -> np.ndarray:
    mean, std = disk_stats(image)
    x = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).to(device).float().div_(255.0)
    z = ((x - mean) / std)[None, None]
    step = max(1, tile - overlap)
    ys = list(range(0, H - tile + 1, step))
    xs = list(range(0, W - tile + 1, step))
    if ys[-1] != H - tile:
        ys.append(H - tile)
    if xs[-1] != W - tile:
        xs.append(W - tile)
    accumulator = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    views = [(0, False), (0, True), (2, False), (2, True)] if tta >= 4 else [(0, False), (0, True)]
    for y0 in ys:
        for x0 in xs:
            crop = z[:, :, y0 : y0 + tile, x0 : x0 + tile].contiguous()
            accumulator_crop = None
            for k, flip in views:
                inp = crop
                if flip:
                    inp = torch.flip(inp, dims=[3])
                if k:
                    inp = torch.rot90(inp, k, dims=[2, 3])
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model(inp.contiguous())
                prob = torch.sigmoid(out.float())
                if k:
                    prob = torch.rot90(prob, -k, dims=[2, 3])
                if flip:
                    prob = torch.flip(prob, dims=[3])
                accumulator_crop = prob if accumulator_crop is None else accumulator_crop + prob
            prob = (accumulator_crop / len(views))[0, 0].cpu().numpy().astype(np.float32) * 255.0
            accumulator[y0 : y0 + tile, x0 : x0 + tile] += prob
            counts[y0 : y0 + tile, x0 : x0 + tile] += 1.0
    return np.clip(accumulator / np.maximum(counts, 1.0), 0, 255).round().astype(np.uint8)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--out-dir", default="/kaggle/working/sem_crossfold")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--folds-json", default=None)
    parser.add_argument("--target-fold", type=int, default=0, help="fold whose val images are used for cross-fold probs")
    parser.add_argument("--test", action="store_true", help="infer the test set instead of the target fold val")
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--tta", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    started = time.time()
    ensure_packages()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = find_data_root(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"data root {data['root']} -> {out_dir}")

    if args.test:
        images = sorted(data["test_images"].glob("*.jpeg"))
        stems = [path.stem for path in images]
        kind = "test"
    else:
        folds = find_folds(args.folds_json)
        stems = sorted(folds["folds"][str(args.target_fold)])
        images = [data["train_images"] / f"{stem}.jpeg" for stem in stems]
        kind = f"fold{args.target_fold}val"
    if args.limit:
        images = images[: args.limit]
        stems = stems[: args.limit]
    log(f"{kind}: {len(images)} images; ckpts: {len(args.ckpt)}")

    for ckpt_path in args.ckpt:
        path = Path(ckpt_path)
        model = load_model(path, device)
        output = {}
        for number, (stem, image_path) in enumerate(zip(stems, images), 1):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(image_path)
            if image.shape != (H, W):
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
            output[stem] = full_probability_tiled(model, image, device, args.tile, args.overlap, args.tta)
            if number % 20 == 0:
                log(f"{path.stem} {kind} {number}/{len(images)} ({time.time()-started:.0f}s)")
        target = out_dir / f"{path.stem}_on_{kind}.npz"
        np.savez_compressed(target, **output)
        log(f"wrote {target} ({target.stat().st_size/1e6:.2f} MB)")
        del model
        torch.cuda.empty_cache()
    meta = {
        "ckpts": [str(p) for p in args.ckpt],
        "kind": kind,
        "n_images": len(images),
        "tile": args.tile,
        "overlap": args.overlap,
        "tta": args.tta,
        "out_dir": str(out_dir),
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / f"meta_{kind}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"DONE {kind} in {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    main()
