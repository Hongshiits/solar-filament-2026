
"""Consensus soft-label semantic U-Net trainer (prize phase).

Key ideas vs public baselines:
  * target = pixel-wise MEAN of all annotator records (soft consensus) -> matches the
    official pooled-PQ objective which rewards consensus geometry;
  * per-image solar-disk mean/std normalisation (as in the public U-Net baselines);
  * train on native-resolution random crops (default 1024) so thin 1-3 px filaments are
    not down-sampled; inference is fully-convolutional at 2048;
  * group 5-fold split by physical jpeg (harness work/cache/folds.json);
  * val: full-resolution fold-held-out inference + official PQ + KEEP rule.

Usage (cloud):
  python src/train_semantic.py --fold 0 --encoder tu-tf_efficientnetv2_s --epochs 40 \
      --crop 1024 --batch 8 --out-dir /kaggle/working/sem_fold0 --device cuda:0
Outputs: best.pt, val_report.json, oof_probs.npz (uint8 p*255 for val fold), train_log.txt
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, random
from pathlib import Path
import numpy as np
import cv2

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMG = DATA / "train" / "train_images"
TRAIN_JSON = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
FOLDS = BASE / "work" / "cache" / "folds.json"
H = W = 2048
DISK_R = 950
KEEP = dict(t_lo=0.35, t_hi=0.99, min_area=200, min_meanp=0.85)

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ----------------------------- data -----------------------------
def load_annotations():
    coco = json.load(open(TRAIN_JSON, encoding="utf-8"))
    by_stem = {}
    for im in coco["images"]:
        annot, stem = im["id"].split("-", 1)
        by_stem.setdefault(stem, []).append(im["id"])
    ann_by_img = {}
    from collections import defaultdict
    tmp = defaultdict(list)
    for a in coco["annotations"]:
        tmp[a["image_id"]].append(a)
    return by_stem, tmp

def rasterize_polygons(seg_list, height=H, width=W):
    from pycocotools import mask as mask_utils
    acc = np.zeros((height, width), dtype=np.uint8)
    for seg in seg_list:
        polys = seg if (len(seg) and isinstance(seg[0], (list, tuple))) else [seg]
        rle = mask_utils.frPyObjects(polys, height, width)
        m = mask_utils.decode(mask_utils.merge(rle))
        acc |= (m > 0).astype(np.uint8) if m.ndim == 2 else (m.max(2) > 0).astype(np.uint8)
    return acc

def build_soft_targets(stems, ann_by_img, by_stem):
    """stem -> soft float32 target (mean over annotator records of that physical image)."""
    out = {}
    for stem in stems:
        recs = by_stem[stem]
        acc = np.zeros((H, W), dtype=np.float32)
        for rid in recs:
            m = rasterize_polygons([a["segmentation"] for a in ann_by_img[rid]])
            acc += m
        out[stem] = np.clip(acc / max(len(recs), 1) * 255.0, 0, 255).round().astype(np.uint8)  # quantized, saves RAM
    return out

YY, XX = np.ogrid[:H, :W]
DISK = ((YY - (H - 1) / 2.0) ** 2 + (XX - (W - 1) / 2.0) ** 2) <= DISK_R ** 2

def disk_stats(img):
    v = img[DISK].astype(np.float32) / 255.0
    return float(v.mean()), max(float(v.std()), 1e-3)

def augment(img_u8, tgt, rng, crop):
    # random crop biased to contain target
    h, w = img_u8.shape
    if crop >= h:
        y0 = x0 = 0
    else:
        if tgt is not None and rng.random() < 0.8:
            ys, xs = np.nonzero(tgt > 0.2)
            if len(ys):
                cy = int(rng.choice(ys)); cx = int(rng.choice(xs))
            else:
                cy, cx = rng.integers(0, h), rng.integers(0, w)
        else:
            cy, cx = rng.integers(0, h), rng.integers(0, w)
        y0 = int(np.clip(cy - crop // 2, 0, h - crop))
        x0 = int(np.clip(cx - crop // 2, 0, w - crop))
    img = img_u8[y0:y0 + crop, x0:x0 + crop]
    t = tgt[y0:y0 + crop, x0:x0 + crop] if tgt is not None else None
    # geometric aug: flips + rot90
    if rng.random() < 0.5: img = img[:, ::-1]; t = t[:, ::-1] if t is not None else None
    if rng.random() < 0.5: img = img[::-1, :]; t = t[::-1, :] if t is not None else None
    k = int(rng.integers(0, 4))
    if k: img = np.rot90(img, k); t = np.rot90(t, k) if t is not None else None
    img = np.ascontiguousarray(img); t = np.ascontiguousarray(t) if t is not None else None
    # photometric aug (gamma / contrast / noise)
    if rng.random() < 0.7:
        g = rng.uniform(0.8, 1.25)
        img = np.clip((img.astype(np.float32) / 255.0) ** g * 255.0, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        c = rng.uniform(0.85, 1.15); b = rng.uniform(-10, 10)
        img = np.clip(img.astype(np.float32) * c + b, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        noise = rng.normal(0, 3.0, size=img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, t




def full_probability(model, img_u8, mean, std, device, tta=2, torch=None):
    torch_ = torch
    x = torch_.from_numpy(np.array(img_u8, dtype=np.uint8, copy=True)).to(device).float().div_(255.0)
    z = ((x - float(mean)) / float(std))[None, None]
    views = [(0, False), (0, True), (2, False), (2, True)] if tta >= 4 else [(0, False), (0, True)]
    acc = None
    for k, flip in views:
        inp = z
        if flip:
            inp = torch_.flip(inp, dims=[3])
        if k:
            inp = torch_.rot90(inp, k, dims=[2, 3])
        inp = inp.contiguous()
        with torch_.autocast(device_type=device.type, dtype=torch_.float16, enabled=device.type == "cuda"):
            out = model(inp)
        prob = torch_.sigmoid(out.float())
        if k:
            prob = torch_.rot90(prob, -k, dims=[2, 3])
        if flip:
            prob = torch_.flip(prob, dims=[3])
        acc = prob if acc is None else acc + prob
    acc = acc / len(views)
    return (acc[0, 0] * 255.0).round_().clamp_(0, 255).to(torch_.uint8).cpu().numpy()

def probs_to_rle(p_u8, keep, stem):
    from pycocotools import mask as mask_utils
    from scipy import ndimage as ndi
    t_lo = int(round(keep["t_lo"] * 255)); t_hi = int(round(keep["t_hi"] * 255))
    lab, n = ndi.label(p_u8 > t_lo, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return []
    lf = lab.ravel(); nz = np.flatnonzero(lf)
    lv, pv = lf[nz], p_u8.ravel()[nz]
    area = np.bincount(lv, minlength=n + 1)[1:].astype(np.int64)
    psum = np.bincount(lv, weights=pv, minlength=n + 1)[1:]
    pmax = np.zeros(n + 1, np.uint8); np.maximum.at(pmax, lv, pv)
    meanp = psum / np.maximum(area, 1) / 255.0
    ok = (pmax[1:] > t_hi) & (area >= keep["min_area"]) & (meanp >= keep["min_meanp"])
    labels = (np.flatnonzero(ok) + 1).tolist()
    objs = ndi.find_objects(lab)
    rows = []
    for k, c in enumerate(labels, 1):
        sl = objs[c - 1]
        m = (lab[sl] == c)
        full = np.zeros((H, W), np.uint8)
        full[sl] = m
        counts = mask_utils.encode(np.asfortranarray(full))["counts"]
        rows.append((f"{stem}_{k}", counts.decode("ascii") if isinstance(counts, bytes) else str(counts)))
    return rows


_G = {}

class SoftCropDataset:
    def __init__(self, stems, crops_per_epoch=1):
        self.stems = list(stems)
        self.repeat = max(1, int(crops_per_epoch))
    def __len__(self):
        return len(self.stems) * self.repeat
    def __getitem__(self, i):
        import torch
        stem = self.stems[i % len(self.stems)]
        rng = np.random.default_rng((i * 1_000_003 + random.randint(0, 10**6)) % (2**32))
        img = cv2.imread(str(Path(_G["img_dir"]) / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        tgt = _G["soft"][stem]
        img_c, tgt_c = augment(img, tgt, rng, _G["crop"])
        mean, std = _G["stats"][stem]
        x = (img_c.astype(np.float32) / 255.0 - mean) / std
        x = np.ascontiguousarray(x[None])
        y = np.ascontiguousarray(tgt_c.astype(np.float32) / 255.0)[None]
        return torch.from_numpy(x), torch.from_numpy(y)


def _soft_erode(x):
    import torch.nn.functional as F
    return -F.max_pool2d(-x, 3, stride=1, padding=1)
def _soft_dilate(x):
    import torch.nn.functional as F
    return F.max_pool2d(x, 3, stride=1, padding=1)
def _soft_open(x):
    return _soft_dilate(_soft_erode(x))
def soft_skeleton(x, iters=8):
    import torch
    x1 = _soft_open(x)
    skel = torch.relu(x - x1)
    for _ in range(iters):
        x = _soft_erode(x)
        x1 = _soft_open(x)
        delta = torch.relu(x - x1)
        skel = skel + torch.relu(delta - skel * delta)
    return skel
def cl_dice_loss(pred_prob, target, smooth=1.0):
    """centerline Dice (Shit et al. 2021) for thin curvilinear structures."""
    skel_pred = soft_skeleton(pred_prob)
    skel_true = soft_skeleton(target)
    tprec = (skel_pred * target).sum() / (skel_pred.sum() + smooth)
    tsens = (skel_true * pred_prob).sum() / (skel_true.sum() + smooth)
    return 1.0 - (2.0 * tprec * tsens) / (tprec + tsens + smooth)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--encoder", default="tu-tf_efficientnetv2_s")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--folds-json", default=None)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--val-images", type=int, default=0, help="debug: cap val images")
    ap.add_argument("--limit-train", type=int, default=0, help="debug: cap train images")
    ap.add_argument("--tta", type=int, default=2)
    ap.add_argument("--workers", type=int, default=(0 if os.name == "nt" else 4))
    ap.add_argument("--crops-per-epoch", type=int, default=4)
    ap.add_argument("--loss", default="bcedice", choices=["bcedice", "bcedice_cldice"])
    ap.add_argument("--cl-dice-weight", type=float, default=0.25)
    args = ap.parse_args()

    import torch, torch.nn as nn
    import segmentation_models_pytorch as smp
    from torch.utils.data import Dataset, DataLoader

    data_root = Path(args.data_root) if args.data_root else DATA
    train_img = data_root / "train" / "train_images"
    train_json = data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    out_dir = Path(args.out_dir) if args.out_dir else (BASE / "work" / f"sem_{args.encoder.split('_')[-1]}_fold{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"start fold={args.fold} encoder={args.encoder} out={out_dir}")

    coco = json.load(open(train_json, encoding="utf-8"))
    from collections import defaultdict
    by_stem = {}
    for im in coco["images"]:
        annot, stem = im["id"].split("-", 1)
        by_stem.setdefault(stem, []).append(im["id"])
    ann_by_img = defaultdict(list)
    for a in coco["annotations"]:
        ann_by_img[a["image_id"]].append(a)
    folds_path = Path(args.folds_json) if args.folds_json else (BASE / "work" / "cache" / "folds.json")
    folds = json.load(open(folds_path, encoding="utf-8"))
    fold_of = folds["fold_of"]
    train_stems = sorted([s for s, f in fold_of.items() if f != args.fold])
    val_stems = sorted(folds["folds"][str(args.fold)])
    if args.limit_train:
        train_stems = train_stems[: args.limit_train]
    if args.val_images:
        val_stems = val_stems[: args.val_images]
    log(f"train {len(train_stems)} / val {len(val_stems)}")

    # drop empty images (no annotations)
    train_stems = [s for s in train_stems if len(by_stem.get(s, [])) > 0]

    # precompute per-image mean/std (disk stats) cache
    stats_cache = out_dir / "disk_stats.json"
    if stats_cache.exists():
        stats = {k: tuple(v) for k, v in json.load(open(stats_cache)).items()}
    else:
        stats = {}
    for s in train_stems + [v for v in val_stems if v not in stats]:
        if s in stats:
            continue
        img = cv2.imread(str(train_img / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
        stats[s] = disk_stats(img)
    json.dump({k: list(v) for k, v in stats.items()}, open(stats_cache, "w"))

    soft = build_soft_targets(train_stems + val_stems, ann_by_img, by_stem)  # uint8 quantized soft consensus targets
    log("soft targets ready")

    _G["img_dir"] = train_img
    _G["soft"] = soft
    _G["stats"] = stats
    _G["crop"] = args.crop
    _G["train_img"] = str(train_img)

    model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=1, classes=1).to(args.device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"model params {n_par/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    bce = nn.BCEWithLogitsLoss()
    loader = DataLoader(SoftCropDataset(train_stems, args.crops_per_epoch), batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=(args.workers > 0))
    steps = max(1, len(loader))
    amp = args.device.startswith("cuda")

    # harness metric import (optional)
    sys.path.insert(0, str(BASE / "src"))
    try:
        import pq_metric as P
    except Exception:
        P = None
    gt_val = None
    if P is not None and val_stems:
        try:
            gt_all = P.gt_df_from_coco(train_json)
            gt_val = P.gt_df_for_stems(gt_all, val_stems)
        except Exception:
            gt_val = None

    def validate(epoch):
        model.eval()
        maps = {}
        with torch.no_grad():
            for j, stem in enumerate(val_stems, 1):
                img = cv2.imread(str(train_img / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
                mean, std = stats[stem]
                maps[stem] = full_probability(model, img, mean, std, torch.device(args.device), tta=args.tta, torch=torch)
                if j % 20 == 0:
                    log(f"  val {j}/{len(val_stems)}")
        # keep probability maps only for the best epoch (disk friendly); stored after scoring below
        if gt_val is not None:
            rows = []
            for stem, p in maps.items():
                rows.extend(probs_to_rle(p, KEEP, stem))
            import pandas as pd
            pred = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
            res = P.evaluate(gt_val, pred)
            log(f"val epoch {epoch}: PQ={res['pq']:.4f} SQ={res['sq']:.4f} RQ={res['rq']:.4f} TP={res['tp']} FP={res['fp']} FN={res['fn']}")
            return float(res["pq"]), res, maps
        # fallback: pixel dice at 0.5 on GT union
        dices = []
        for stem, p in maps.items():
            recs = by_stem.get(stem, [])
            acc = np.zeros((H, W), np.float32)
            for rid in recs:
                acc += rasterize_polygons([a["segmentation"] for a in ann_by_img[rid]])
            gt = (acc / max(len(recs), 1)) > 0.5
            pr = p > int(0.5 * 255)
            inter = np.logical_and(gt, pr).sum(); den = gt.sum() + pr.sum()
            dices.append(2 * inter / max(den, 1))
        return float(np.mean(dices)), {"dice": float(np.mean(dices))}, maps

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(args.device, non_blocking=True); y = y.to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss_bce = bce(logits, y)
                prob = torch.sigmoid(logits)
                num = 2 * (prob * y).sum()
                den = prob.sum() + y.sum() + 1.0
                loss_dice = 1 - num / den
                loss = 0.5 * loss_bce + 0.5 * loss_dice
                if args.loss == "bcedice_cldice":
                    loss = loss + args.cl_dice_weight * cl_dice_loss(prob, y)
            opt.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        log(f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} lr={sched.get_last_lr()[0]:.2e} elapsed={(time.time()-t0)/60:.1f}min")
        if epoch % args.val_every == 0 or epoch == args.epochs:
            score, res, maps = validate(epoch)
            torch.save({"epoch": epoch, "model": model.state_dict(), "encoder": args.encoder, "val": res}, out_dir / "last.pt")
            if score > best:
                best = score
                torch.save({"epoch": epoch, "model": model.state_dict(), "encoder": args.encoder, "val": res}, out_dir / "best.pt")
                np.savez_compressed(out_dir / "val_probs_best.npz", **{k: v for k, v in maps.items()})
                log(f"  new best {score:.4f}")
                def _jsonable(obj):
                    if isinstance(obj, dict):
                        return {k: _jsonable(v) for k, v in obj.items() if k not in ("per_image", "tp_ious")}
                    if hasattr(obj, "tolist"):
                        return obj.tolist()
                    return obj
                json.dump({"fold": args.fold, "encoder": args.encoder, "loss": args.loss, "epoch": epoch, "score": score,
                           "val": _jsonable(res)},
                          open(out_dir / "val_report.json", "w"), indent=1, default=float)
    log(f"done best={best:.4f} out={out_dir}")

if __name__ == "__main__":
    main()