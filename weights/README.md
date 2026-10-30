# Model weights

The notebook **loads pre-trained weights from this directory** and does not retrain
anything, so the reported predictions can be reproduced directly (explicit evaluation
requirement: "the model's weights must be stored in the repo, and be loaded in the notebook").

## Files

| file | size | MD5 | SHA256 | description |
|---|---|---|---|---|
| `best_det.pt` | 20,591,908 B (19.6 MiB) | `fd91ebd028e0ab58d7ff726013104c92` | `fd3f534f770b7027ba900bd4d96b3559e5888b612868ec86eaf3ae16b8396814` | YOLO11-seg detector (single class "filament"), 1024 px input, trained on all 1154 official annotator records of the official train split |
| `best_ref.pt` | 57,052,815 B (54.4 MiB) | `678a319cb4bd58d507ea89ea3b1d3a7f` | `581280342f1ee04ea0d0cef6f8fceb449264392e5c9fe737e03f74894e618409` | seed-conditioned ResNet18 U-Net crop refiner, 256 px crops, 3 input channels [gray, CLAHE(gray), seed], BCE + soft-Dice |

Total: 77,644,723 B (74.0 MiB) — below GitHub's 100 MB per-file limit, so **the weights are
committed directly in the repository (no Git LFS needed)**.

Verify with:

    python scripts/check_weights.py

Expected output: `[OK]` for both files.

## Training configuration (summary)

- Detector: `src/train_yolo_seg.py` — YOLO11s-seg, 1024 px, 30 epochs, batch 8 (effective),
  AdamW (ultralytics auto), single class, medoid/all-annotator targets;
  training on our grouped fold split (fold-0 model, 565 physical images / 922 annotator records).
- Refiner: `src/train_refiner.py` — 256 px crops cut with context 1.8, 8 epochs, AdamW 3e-4,
  BCE + soft-Dice, 4-flip TTA at inference with a topology guard.
- Thresholds were selected with the official pooled-PQ replica (`src/pq_metric.py`) on the
  fold-0 validation split: conf 0.35 / refiner 0.50 / TTA "guard".
- Full cloud training run (Kaggle T4, one fold): 89 min; fold-0 validation PQ = 0.3780
  (SQ 0.6475 / RQ 0.5838), test submission: 1065 instances on 177/180 images
  (see `../docs/report.pdf` and `../REPO_CHECK.md`).

## Provenance and licence

- Both checkpoints were trained by this team on the **official competition data only**
  (MAGFiLO 1.0 / GONG H-alpha, CC BY-NC 4.0), from ImageNet/COCO-pretrained backbones.
  No test ground truth, no public-MAGFiLO test annotations, no external filament labels.
- Files are released for non-commercial research/reproduction only (see `../LICENSE` and
  `../docs/DATA_AND_LICENSES.md`).
