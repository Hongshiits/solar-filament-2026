# Solar Filament Segmentation Challenge 2026 — Reproducible Solution

Team: **Tooku (Kaggle: ptcuoigl; contact: hongshiite@outlook.com, Hunan, China)** · Competition: [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026) · Task: instance segmentation of solar filaments in GONG H-alpha full-disk images (2048×2048).
Official metric: instance-level **Panoptic Quality (PQ)** with IoU > 0.5 matching, scored per annotator pass.

> This repository is the reference implementation submitted for the final evaluation of the competition.
> The pipeline runs on the official test images only; no test ground truth, no external filament annotations and no future/neighbouring frames are used at inference time.
> Training uses only the official MAGFiLO annotations and images shipped with the competition. Public pre-trained weights that we reproduce were trained by their authors on that same official split; their provenance is documented in weights/README.md. No external filament annotations and no public-MAGFiLO test labels are used.

## 1. Method at a glance

Two-stage instance-segmentation cascade, followed by metric-aware assembly:

1. **Detector** — YOLO11-seg (single class "filament") on H-alpha images resized to 1024×1024, initialised from COCO weights. It outputs one coarse mask + confidence per candidate filament.
2. **Seed-conditioned crop refiner** — for every candidate, a square crop of side ceil(1.8 × max(bbox side)) (min 96 px) is cut from the full-resolution image, resized to 256×256, and fed to a ResNet18 U-Net with 3 channels [gray, CLAHE(gray), seed mask]. The refiner predicts a per-pixel filament probability; the threshold and the minimum area are selected on out-of-fold PQ.
3. **Test-time augmentation** — four flips (identity/hflip/vflip/hvflip) with a topology guard (area ratio 0.6–1.7, Dice ≥ 0.62, no extra components); falls back to single view when the guard fails.
4. **Pixel-disjoint assembly** — predictions are sorted by score and greedily allocated so that no two instances share a pixel (the official scorer rejects overlapping predictions); residual fragments below 20–96 px are dropped. An optional five-fold ensemble clusters predictions at mask IoU ≥ 0.5 and keeps clusters supported by at least two folds.
5. **Encoding** — COCO compressed RLE on 2048×2048 masks; one CSV row per filament with filament_id = "<image_stem>_<k>".

Full hyper-parameters are in the report (docs/report.pdf) and in configs/*.yaml.

## 2. Verified results

| Model / submission | Public LB (PQ) | Notes |
|---|---|---|
| Public Mask R-CNN + U-Net refiner reproduction (ref 57763686) | **0.34** | byte-level reproduction of the public "0.70" inference handoff; that 0.70 was a pre-August **Dice-era** score and is not comparable to the current PQ leaderboard |
| **Our YOLO11-seg + crop-refiner cascade, fold 0** (ref 57768688) | **0.34** | self-trained model in this repository; fold-0 validation PQ = 0.3780 (official pooled replica, SQ 0.6475 / RQ 0.5838); 1065 instances on 177/180 images |
| Public dariush U-Net ensemble, 2 models, TTA2 (ref 57767915) | **0.37** | semantic U-Net ensemble reproduced from the public weights dataset; format-validation submission |
| **Public dariush U-Net ensemble, 10 models, TTA8 - final selected submission** (ref 57768486) | **0.39** | 1177 instances on 177/180 images; structure identical to the author's published submission; both submissions are reproducible from public assets documented in the report |
| Human annotator vs annotator (reference) | 0.3399 (pooled PQ) | 296 multi-annotator training observations, official scorer |
| Public leaderboard position at freeze (2026-10-29) | **248 / 1260 teams** | public half of the test set; final ranking uses the private half + jury review |
| Consensus-target oracle (upper bound) | 0.6547 | majority-vote consensus mask scored against all annotation passes |

**Final-evaluation requirements** (host instructions): a 4-page technical report (docs/report.pdf, official
Overleaf template), a public code repository with requirements.txt, a complete pipeline notebook and the
trained weights committed in the repository (this repository), and the organizer Google Form submitted
before Nov 15, 2026 23:59 CDT. The jury evaluates 70% quantitative + 30% qualitative; the leaderboard is
cleaned after the competition.

## 3. Repository structure

    .
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
    ├── configs/                  # training / inference configs (seeds, thresholds)
    ├── notebooks/
    │   └── final_pipeline.ipynb # loads weights → inference on 180 test images → submission.csv
    ├── src/                      # reusable modules (data, folds, models, metric, validation)
    ├── scripts/
    │   ├── check_weights.py      # verifies MD5/SHA256 of the weight files
    │   ├── download_weights.py   # weights are committed directly; verifies presence
    │   └── smoke_inference.py    # CPU smoke test of the notebook inference path
    ├── weights/
    │   └── README.md             # weight files and how to obtain them
    ├── REPO_CHECK.md            # publication checklist, hashes, verification record
    └── docs/
        ├── DATA_AND_LICENSES.md
        ├── TRAIN.md              # training protocol / hyper-parameters
        ├── INFERENCE.md          # inference contract (inputs, outputs, cost)
        └── report.pdf            # technical report (official template)
    figures/                      # report figures (vector PDF + 300 dpi PNG) and make_figs.py

## 4. Environment

Tested with Python 3.14; local development on one NVIDIA RTX 2080 Super Max-Q (8 GB), cloud training and inference on one NVIDIA T4.

    # 1) CUDA build of PyTorch first (adjust cuXXX to your driver)
    pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128

    # 2) everything else
    pip install -r requirements.txt

## 5. Reproduce the submission (inference only, no training)

Expected runtime: about **1.5 minutes** for all 180 test images on one NVIDIA T4 (0.5 s/image).

    # 1) get the code and the weights
    git clone https://github.com/Hongshiits/solar-filament-2026 solar-filament
    cd solar-filament
    ./scripts/check_weights.py            # or: python scripts/check_weights.py

    # 2) place the competition data (official Kaggle download, unzipped) next to the repo
    #    data/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg
    python -c "import yaml; print(yaml.safe_load(open('configs/inference.yaml'))['data_root'])"

    # 3) run the notebook end-to-end (produces submission.csv)
    jupyter nbconvert --to notebook --execute notebooks/final_pipeline.ipynb --output final_pipeline_out.ipynb

    # 4) validate the file before uploading
    python -m src.validate_submission submission.csv

The notebook performs, in order: environment check → weight checksum → model load → inference over the 180 test images → pixel-disjoint assembly → CSV encoding → local validation (schema, image coverage, RLE decodability, per-image overlap check, positive area).

## 6. Reproduce training (optional, GPU recommended)

    python -m src.data_prep        --data-root data/MAGFiLO_1.0_Kaggle_2026 --cache work/cache
    python -m src.splits           --cache work/cache --n-folds 5 --seed 42
    python -m src.train_yolo_seg   data  --cache work/cache
    python -m src.train_yolo_seg   train --fold 0 --imgsz 1024 --epochs 30
    python -m src.train_refiner    train --fold 0 --crop 256 --epochs 8
    python -m src.infer_modeling   --split val --fold 0        # OOF predictions + PQ
    python -m src.infer_modeling   --split test                # final submission.csv

Training used **grouped 5-fold cross-validation, split by physical JPEG** (the same observation annotated by several people is never split across folds), training target = every annotation record of a physical image (each of the 1154 records is a separate training sample, so the model learns the consensus geometry), and evaluation against **every** annotator pass.

GPU-hours actually used: 0.91 for the detector and 0.15 for the refiner on one NVIDIA T4 (about 1.5 GPU-h including validation sweeps and test inference).

## 7. Weights

See weights/README.md. The model weights are stored as **GitHub Release assets** (tag `weights-v1`) with Git LFS as an alternative; the notebook loads them directly, so the reported predictions can be reproduced **without retraining**. MD5 checksums are verified by scripts/check_weights.py.

## 8. Data, licences and competition compliance

- Competition data: MAGFiLO 1.0 / GONG H-alpha, **CC BY-NC 4.0** — non-commercial and academic use only. See docs/DATA_AND_LICENSES.md.
- Code in this repository: **MIT** (see LICENSE).
- Inference uses only the H-alpha test images provided by the competition (rules §6). No external ground truth, no public-MAGFiLO test annotations, no neighbouring frames at inference.
- External data used for training: none beyond the official competition data. Public pre-trained weights that we reproduce were trained by their authors on that same official split and are documented in weights/README.md.
- AI-assisted coding tools were used for coding assistance, experiment scripting and figure generation; all design decisions, experiments and the report were reviewed and can be defended by the team (competition rules §8b).

## 9. Attribution

This work uses GONG data obtained by the NSO Integrated Synoptic Program, managed by the National Solar Observatory (NSO), operated by AURA, Inc. under a cooperative agreement with the National Science Foundation and with contribution from NOAA. The GONG network is hosted by Big Bear Solar Observatory, High Altitude Observatory, Learmonth Solar Observatory, Udaipur Solar Observatory, Instituto de Astrofísica de Canarias, and Cerro Tololo Interamerican Observatory. Ground truth: MAGFiLO (Ahmadzadeh et al., 2024, Scientific Data, doi:10.1038/s41597-024-03876-y). Competition organized by the Earth-Space AI Research (ESAIR) Lab.

## 10. Citation

    @misc{tooku2026filament,
      title  = {A Solution to the Solar Filament Segmentation Challenge 2026},
      author = {Tooku},
      year   = {2026},
      note   = {Kaggle competition, https://www.kaggle.com/competitions/filament-segmentation-2026}
    }

## Additional experiments

Post-baseline iterations (semantic U-Net training, probability/mask fusion, public ensemble reproduction, support pruning) with their honest numbers and negative results are archived in [`experiments/`](experiments/README.md). None of them beat the public 0.39 baseline on the leaderboard; the directory documents what was tried and why it did not help.