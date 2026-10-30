"""Reproduce final submission #1: public 10x U-Net ensemble with 8-view TTA (public LB 0.39).

Execution environment: a Kaggle script kernel (GPU) with two attached inputs:
  - the public weights dataset  https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights
    (contains solar_unet_receipt.json, folds.csv, 10 checkpoints, smp wheel),
  - the competition data (test/test_images).
It writes submission.csv to /kaggle/working. Expected output: 1177 rows over 177 images,
public LB 0.39 (Kaggle ref 57768486).

The inference logic below follows the public author's notebook (receipt-verified); CONFIG selects
the number of models and TTA views.
"""
import glob
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

T0 = time.time()

# =============================================================================
# OUR CONFIG (kaggle-runner wrapper; the inference logic below is the author's)
#   max_models: 0 = all 10 models, N = first N models (quick format validation)
#   tta: number of D4 views per image (1, 2, 4, 8)
#   skip_analysis: skip the author's explanatory analysis tail (not needed for a run)
# =============================================================================
CONFIG = {
    "max_models": 0,   # 0 = all 10 models (full run)
    "tta": 8,          # full 8-view D4 TTA
    "skip_analysis": True,
    "keep": dict(t_lo=0.35, t_hi=0.99, min_area=200, min_meanp=0.85),
}

INPUT, OUT = "/kaggle/input", "/kaggle/working"


def find_file(name):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", name), recursive=True))
    assert len(hits) == 1, f"expected exactly one {name} under {INPUT}, found {hits[:3]}"
    return hits[0]


RECEIPT_PATH = find_file("solar_unet_receipt.json")
WEIGHTS_DIR = os.path.dirname(RECEIPT_PATH)
TEST_DIR = [d for d in sorted(glob.glob(os.path.join(INPUT, "**", "test_images"), recursive=True)) if os.path.isdir(d)]
assert len(TEST_DIR) == 1, TEST_DIR
TEST_DIR = TEST_DIR[0]
print("weights dataset:", WEIGHTS_DIR)
print("competition test images:", TEST_DIR)

try:
    import segmentation_models_pytorch as smp
except ImportError:  # not in the Kaggle image: install the wheel that ships with the weights (no internet needed)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                    find_file("segmentation_models_pytorch-0.5.0-py3-none-any.whl")], check=True)
    importlib.invalidate_caches()
    import segmentation_models_pytorch as smp
import timm
import torch
from PIL import Image
from pycocotools import mask as mask_util
from scipy import ndimage as ndi

print(f"python {sys.version.split()[0]} | torch {torch.__version__} | timm {timm.__version__} | "
      f"segmentation_models_pytorch {smp.__version__} | numpy {np.__version__} | pandas {pd.__version__}")
GPUS = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("GPUs:", GPUS if GPUS else "none, so this will take hours; switch the accelerator to GPU T4 x2")

KEEP = dict(CONFIG["keep"])  # author rule chosen by nested CV
TTA = int(CONFIG["tta"])  # subset of the author 8 views (1, 2, 4 or 8)
H = W = 2048
DISK_R = 950  # px around the image centre: each image is normalised with the statistics inside this circle

with open(RECEIPT_PATH) as f:
    RECEIPT = json.load(f)
MODELS = RECEIPT["ensemble"]["models"]  # file, encoder, fold and sha256 of every weight file
if int(CONFIG["max_models"]) > 0:
    MODELS = MODELS[: int(CONFIG["max_models"])]
print(f"RUN CONFIG: models={len(MODELS)} tta={TTA} skip_analysis={CONFIG['skip_analysis']}")
print(f"{len(MODELS)} models = {len({m['fold'] for m in MODELS})} folds x {len({m['encoder'] for m in MODELS})} "
      f"encoders {sorted({m['encoder'] for m in MODELS})}")
print("keep rule:", KEEP)
print(f"TTA views: {TTA} | image size: {H} x {W} | normalisation disk radius: {DISK_R} px | "
      f"each model's map stored as round(p x 255)")

YY, XX = np.ogrid[:H, :W]
DISK = ((YY - (H - 1) / 2.0) ** 2 + (XX - (W - 1) / 2.0) ** 2) <= DISK_R ** 2


def load_gray(path):
    with Image.open(path) as im:
        assert im.size == (W, H), (path, im.size)
        return np.asarray(im.convert("L"), dtype=np.uint8)


def disk_stats(img):
    v = img[DISK].astype(np.float32) / 255.0
    return v.mean(), max(float(v.std()), 1e-3)


stems = sorted(os.path.basename(p)[:-len(".jpeg")] for p in glob.glob(os.path.join(TEST_DIR, "*.jpeg")))
assert stems and all(re.fullmatch(r"\d{14}[A-Z][a-z]", s) for s in stems), stems[:3]
with ThreadPoolExecutor(8) as ex:
    test_u8 = np.stack(list(ex.map(load_gray, [os.path.join(TEST_DIR, s + ".jpeg") for s in stems])))
stats = np.array([disk_stats(im) for im in test_u8], dtype=np.float32)
print(f"{len(stems)} test images of {H} x {W}, {test_u8.nbytes / 1e9:.2f} GB in memory")
print(f"disk mean of pixel/255 from {stats[:, 0].min():.3f} to {stats[:, 0].max():.3f}, "
      f"disk SD from {stats[:, 1].min():.3f} to {stats[:, 1].max():.3f}")

D4 = [(0, False), (0, True), (2, False), (2, True), (1, False), (1, True), (3, False), (3, True)]  # (rot90 k, flip)


def _tf(z, k, flip):
    if flip:
        z = torch.flip(z, dims=[3])
    return torch.rot90(z, k, dims=[2, 3]) if k else z


def _itf(p, k, flip):
    if k:
        p = torch.rot90(p, -k, dims=[2, 3])
    return torch.flip(p, dims=[3]) if flip else p


@torch.no_grad()
def predict(model, img_u8, mean, std, device, tta=TTA, views_per_pass=4):
    """Full-resolution probability map, averaged over the TTA views, rounded to uint8 (p x 255)."""
    gpu = device.type == "cuda"
    x = torch.from_numpy(np.array(img_u8, dtype=np.uint8, copy=True)).to(device).float().div_(255.0)
    z = ((x - float(mean)) / float(std))[None, None]
    ops = D4[:tta]
    acc = torch.zeros((H, W), dtype=torch.float32, device=device)
    for j in range(0, len(ops), views_per_pass):
        chunk = ops[j:j + views_per_pass]
        inp = torch.cat([_tf(z, k, fl) for k, fl in chunk])
        if gpu:
            inp = inp.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=gpu):
            out = model(inp)
        prob = torch.sigmoid(out.float())
        for q, (k, fl) in enumerate(chunk):
            acc += _itf(prob[q:q + 1], k, fl)[0, 0]
    acc /= len(ops)
    return (acc * 255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(m, device):
    model = smp.Unet(encoder_name=m["encoder"], encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, m["file"]), map_location="cpu", weights_only=True))
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model.eval()


torch.backends.cudnn.benchmark = True
devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]
total = np.zeros((len(stems), H, W), np.uint16)  # sum of the models' uint8 maps
lock = threading.Lock()
todo, done, errors, log = list(MODELS), [], [], []  # EfficientNetV2-S folds first: they are the slower ones


def worker(device):
    while True:
        with lock:
            if not todo or errors:
                return
            m = todo.pop(0)
        try:
            t = time.time()
            ok = sha256(os.path.join(WEIGHTS_DIR, m["file"])) == m["sha256"]
            assert ok, f"{m['file']}: sha256 differs from the receipt"
            model = load_model(m, device)
            for j in range(len(stems)):
                p = predict(model, test_u8[j], stats[j, 0], stats[j, 1], device)
                with lock:
                    total[j] += p
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            with lock:
                done.append(m["file"])
                log.append(f"{m['file']:38s} {str(device):7s} sha256 matches the receipt, "
                           f"{len(stems)} images in {time.time() - t:.0f} s")
        except Exception as e:
            with lock:
                errors.append(f"{m['file']}: {e!r}")


t = time.time()
threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
for th in threads:
    th.start()
for th in threads:
    th.join()
print("\n".join(log))
assert not errors, errors
assert sorted(done) == sorted(m["file"] for m in MODELS), done
avg = np.empty((len(stems), H, W), np.uint8)
for j in range(len(stems)):
    avg[j] = np.clip(np.round(total[j].astype(np.float32) / len(MODELS)), 0, 255).astype(np.uint8)
del total
print(f"{len(done)} models x {len(stems)} images on {len(devices)} device(s) in {(time.time() - t) / 60:.1f} min")

S8 = np.ones((3, 3), dtype=bool)  # 8-connectivity


def u8(t):
    """Probability threshold on the uint8 map: the rule everywhere is p_u8 > u8(t)."""
    return int(round(float(t) * 255))


def components(p_u8, t_lo):
    """8-connected regions of p > t_lo with their area, mean p (0..1) and peak p (uint8)."""
    lab, n = ndi.label(p_u8 > u8(t_lo), structure=S8)
    lab_f = lab.ravel()
    nz = np.flatnonzero(lab_f)
    lv, pv = lab_f[nz], p_u8.ravel()[nz]
    area = np.bincount(lv, minlength=n + 1)[1:].astype(np.int64)
    psum = np.bincount(lv, weights=pv, minlength=n + 1)[1:]
    pmax = np.zeros(n + 1, np.uint8)
    np.maximum.at(pmax, lv, pv)
    meanp = psum / np.maximum(area, 1) / 255.0
    return lab, area, meanp, pmax[1:]


def kept(area, meanp, pmax, rule):
    ok = (pmax > u8(rule["t_hi"])) & (area >= rule["min_area"]) & (meanp >= rule["min_meanp"])
    return np.flatnonzero(ok) + 1  # component labels


def encode(lab, labels):
    objs = ndi.find_objects(lab)
    out = []
    for c in labels:
        sl = objs[c - 1]
        m = np.zeros((H, W), np.uint8)
        m[sl] = lab[sl] == c
        out.append(mask_util.encode(np.asfortranarray(m))["counts"].decode("ascii"))
    return out


t = time.time()
rows = []
for j, stem in enumerate(stems):
    lab, area, meanp, pmax = components(avg[j], KEEP["t_lo"])
    rows += [(f"{stem}_{k + 1}", rle) for k, rle in enumerate(encode(lab, kept(area, meanp, pmax, KEEP)))]
sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
SUB_PATH = os.path.join(OUT, "submission.csv")
sub.to_csv(SUB_PATH, index=False)
n_img = sub.filament_id.str.split("_", n=1).str[0].nunique()
print(f"{len(sub)} filaments on {n_img} of {len(stems)} images, {len(sub) / len(stems):.2f} per test image, "
      f"written to {SUB_PATH} in {time.time() - t:.0f} s")

ID_RE = re.compile(r"^(\d{14}[A-Z][a-z])_(\d+)$")
CHECKS = ["header_ok", "no_quotes_in_file", "ascii_only", "ids_match_pattern", "all_stems_in_test_set",
          "stems_without_underscore", "ids_unique", "rle_charset_ok", "decode_2048_ok", "roundtrip_identical",
          "no_empty_masks", "pixel_disjoint_dense", "pixel_disjoint_rle_iou", "pandas_reread_identical"]


def rles(strs):
    return [{"size": [H, W], "counts": s.encode("ascii")} for s in strs]


def disjoint_by_iou(df):
    stems_ = df["filament_id"].str.split("_", n=1).str[0]
    for stem, g in df.groupby(stems_):
        if len(g) > 1:
            r = rles(g["segmentation_rle"])
            iou = np.asarray(mask_util.iou(r, r, [0] * len(r)))
            np.fill_diagonal(iou, 0)
            if (iou > 0).any():
                return False
    return True


def validate_submission(path, test_stems):
    raw = open(path, "rb").read()
    lines = raw.decode("ascii", "replace").splitlines()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    c = {"sha256": hashlib.sha256(raw).hexdigest(), "n_rows": len(df)}
    c["header_ok"] = lines[0] == "filament_id,segmentation_rle" and list(df.columns) == ["filament_id",
                                                                                          "segmentation_rle"]
    c["no_quotes_in_file"] = b'"' not in raw
    c["ascii_only"] = all(b < 128 for b in raw)
    stems_ = df["filament_id"].str.split("_", n=1).str[0]
    c["ids_match_pattern"] = bool(df["filament_id"].str.match(ID_RE.pattern).all())
    c["all_stems_in_test_set"] = bool(stems_.isin(set(test_stems)).all())
    c["stems_without_underscore"] = bool((~stems_.str.contains("_")).all())
    c["ids_unique"] = bool(df["filament_id"].is_unique)
    c["rle_charset_ok"] = bool(df["segmentation_rle"].map(lambda s: len(s) > 0 and all(48 <= ord(ch) <= 111
                                                                                        for ch in s)).all())
    bad_decode = bad_roundtrip = empty = overlap = 0
    for stem, g in df.groupby(stems_):
        acc = np.zeros((H, W), np.uint8)
        for s in g["segmentation_rle"]:
            try:
                mk = mask_util.decode({"size": [H, W], "counts": s.encode("ascii")})
            except Exception:
                bad_decode += 1
                continue
            if mk.shape != (H, W):
                bad_decode += 1
                continue
            empty += int(mk.sum() == 0)
            bad_roundtrip += int(mask_util.encode(np.asfortranarray(mk))["counts"].decode("ascii") != s)
            acc += mk
        overlap += int(acc.max() > 1)
    c["decode_2048_ok"], c["roundtrip_identical"] = bad_decode == 0, bad_roundtrip == 0
    c["no_empty_masks"], c["pixel_disjoint_dense"] = empty == 0, overlap == 0
    c["pixel_disjoint_rle_iou"] = disjoint_by_iou(df)
    c["pandas_reread_identical"] = bool(pd.read_csv(path, dtype=str, keep_default_na=False).equals(df))
    c["n_images_with_rows"] = int(stems_.nunique())
    c["all_pass"] = all(c[k] for k in CHECKS) and len(df) > 0
    return c


chk = validate_submission(SUB_PATH, stems)
print(f"{sum(chk[k] for k in CHECKS)} of {len(CHECKS)} checks pass:")
for k in CHECKS:
    print(f"  {k:26s} {chk[k]}")
assert chk["all_pass"], "the file failed a check: do not submit it"
N_TEST = 180  # images in the competition's test set
ref = RECEIPT["runs"]["v6"]["submission_file"]
print(f"complete test set: {len(stems) == N_TEST} ({len(stems)} of {N_TEST} images)")
print(f"this run : {chk['n_rows']} rows, {chk['n_images_with_rows']} images with rows, sha256 {chk['sha256'][:16]}")
print(f"submitted: {ref['rows']} rows, {ref['images_with_rows']} images with rows, sha256 {ref['sha256'][:16]}")
print("byte-identical to the file I submitted:", chk["sha256"] == ref["sha256"])

if CONFIG["skip_analysis"]:
    print("analysis tail skipped (CONFIG[skip_analysis]=True)")
    raise SystemExit(0)


def pq_sparse(o, r):
    """Expected PQ when each labelled filament is kept with probability r (o = pooled OOF counts)."""
    return r * o["sum_iou"] / (r * o["tp"] + 0.5 * o["fp"] + 0.5 * (1 - r) * o["tp"] + 0.5 * r * o["fn"])


def solve_r(o, target):  # PQ' rises with r, so bisect
    lo, hi = 0.01, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pq_sparse(o, mid) < target else (lo, mid)
    return (lo + hi) / 2


runs = RECEIPT["runs"]
names = {"v3": "5 x ResNet34", "v5": "5 x EfficientNetV2-S", "v6": "all 10 (this notebook)"}
R = RECEIPT["sparse_label_model"]["r"]
print(f"{'models':24s} {'nested OOF PQ':>13s} {'public':>7s} {'PQ at r = ' + str(R):>15s}")
for v, name in names.items():
    o = runs[v]["oof"]
    print(f"{name:24s} {o['nested_cv_pq']:13.4f} {runs[v]['public_lb']:7.2f} {pq_sparse(o, R):15.4f}")
print(f"gap between nested OOF PQ and public: "
      + ", ".join(f"{runs[v]['oof']['nested_cv_pq'] - runs[v]['public_lb']:.2f}" for v in names))

pub3 = runs["v3"]["public_lb"]
r_lo, r_hi = (solve_r(runs["v3"]["oof"], pub3 + d) for d in (-0.005, 0.005))
print(f"\nr that turns my first model's OOF counts into a score shown as {pub3:.2f} ({pub3 - 0.005:.3f} to "
      f"{pub3 + 0.005:.3f}): {r_lo:.3f} to {r_hi:.3f} (point fit {solve_r(runs['v3']['oof'], pub3):.2f})")
for v in ("v5", "v6"):
    a, b = (pq_sparse(runs[v]["oof"], r) for r in (r_lo, r_hi))
    shown = sorted({f"{x:.2f}" for x in (a, b)})
    print(f"predicted public for {names[v]}: {a:.3f} to {b:.3f}, shown as {' or '.join(shown)}; "
          f"actual {runs[v]['public_lb']:.2f}")

od = RECEIPT["oof_data"]
print(f"\na training annotator outlines {od['gt_per_set']:.2f} filaments per image on average ({od['images']} images, "
      f"{od['annotator_sets']:,} annotator-sets); at r = {R} a test annotator would outline about "
      f"{od['gt_per_set'] * R:.2f}")
print(f"the models predict {runs['v6']['oof']['pred_per_image']:.2f} filaments per training image "
      f"out-of-fold and {len(sub) / len(stems):.2f} per test image in this run")
hh = RECEIPT["human_vs_human_pq"]
print(f"for scale: one expert scored against another gets PQ {hh['value']:.4f} ({hh['value']:.2f}) on the "
      f"{hh['images']} training images with two or three annotators")
o6 = runs["v6"]["oof"]
print(f"this notebook's models: nested OOF PQ {o6['nested_cv_pq']:.2f}, public {runs['v6']['public_lb']:.2f}, "
      f"PQ' {pq_sparse(o6, R):.2f}")

grid = pd.read_csv(find_file("oof_keep_rule_grid.csv"))
grid["pq_sparse"] = [pq_sparse({"sum_iou": s * tp, "tp": tp, "fp": fp, "fn": fn}, R)
                     for s, tp, fp, fn in zip(grid.sq, grid.tp, grid.fp, grid.fn)]
rules = {"KEEP (this notebook)": KEEP,
         "my first model's rule": runs["v3"]["keep_rule"],
         "the EfficientNetV2-S rule": runs["v5"]["keep_rule"],
         "loose: peak > 0.5, no mean filter": dict(t_lo=0.35, t_hi=0.5, min_area=250, min_meanp=0.0)}
tables = {}
print(f"{'rule':34s} {'t_lo':>5s} {'t_hi':>6s} {'area':>5s} {'mean':>5s} {'OOF PQ':>7s} {'PQ at r':>8s} "
      f"{'per test image':>15s}")
for name, rule in rules.items():
    hit = grid[(grid.t_lo == rule["t_lo"]) & (grid.t_hi == rule["t_hi"]) & (grid.min_area == rule["min_area"])
               & (grid.min_meanp == rule["min_meanp"])]
    n = 0
    for j in range(len(stems)):
        if (j, rule["t_lo"]) not in tables:
            tables[(j, rule["t_lo"])] = components(avg[j], rule["t_lo"])[1:]
        n += len(kept(*tables[(j, rule["t_lo"])], rule))
    oof = f"{hit.pq.iloc[0]:7.4f} {hit.pq_sparse.iloc[0]:8.4f}" if len(hit) else f"{'not in the grid':>16s}"
    print(f"{name:34s} {rule['t_lo']:5g} {rule['t_hi']:6g} {int(rule['min_area']):5d} {rule['min_meanp']:5.2f} {oof} "
          f"{n / len(stems):15.2f}")
b = grid.loc[grid.pq_sparse.idxmax()]
print(f"\n{len(grid):,} rules in the grid. Best by OOF PQ: {grid.pq.max():.4f}. Best by PQ at r = {R}: "
      f"{b.pq_sparse:.4f} (t_lo {b.t_lo:g}, t_hi {b.t_hi:g}, area {int(b.min_area)}, mean {b.min_meanp:.2f})")

print(f"total run time {(time.time() - T0) / 60:.1f} min; output: {sorted(os.listdir(OUT))}")