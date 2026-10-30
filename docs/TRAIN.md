# Training protocol (how the submitted weights were produced)

Only the official MAGFiLO annotations shipped with the competition are used as supervision. External unlabelled data (if any) is disclosed in the README and the report.

## Data and splits

- unit of sampling: the **physical JPEG**. The 1-3 annotation passes of the same observation always stay in the same fold (grouped split), otherwise the same image would leak between train and validation.
- 5-fold grouped cross-validation, seed 42, stratification by observatory site and number of annotated filaments.
- detector/refiner training target: all official annotator records (LABEL_SOURCE=all), evaluated against every annotator pass (chosen to avoid biasing training towards over-segmentation).
- evaluation during development always scores **every** annotation pass, exactly like the official metric.

## Stage 1 - detector

| item | value |
|---|---|
| model | YOLO11s-seg (ultralytics 8.4.191) |
| initialisation | COCO-pretrained yolo11s-seg.pt |
| input | 1024 x 1024 grayscale -> 3 channels |
| batch / epochs | 8 / 30 |
| optimizer / LR | AdamW (ultralytics auto: lr0=0.002, weight_decay=0.0005, cosine schedule) |
| augmentation | hsv_v 0.25; fliplr/flipud 0.5; scale 0.20; translate 0.05; mosaic 0.25 (close_mosaic 5); no rotations |
| loss | ultralytics segmentation loss (BCE + DFL box loss + mask BCE) |

## Stage 2 - seed-conditioned crop refiner

| item | value |
|---|---|
| crop | context 1.8 x max(bbox side), min 96 px, resized to 256 |
| input channels | grayscale, CLAHE, seed mask |
| architecture | ResNet18 encoder + U-Net decoder, 3 input channels [gray, CLAHE(gray), seed], 1 output channel |
| batch / epochs | 16 / 8 |
| optimizer / LR | AdamW lr 3e-4, weight_decay 1e-4, cosine schedule |
| augmentation | 90-degree rotations, flips, zoom 0.85-1.15 (pad/crop), brightness gain 0.9-1.1, synthetic seed (GT / coarse / box variants) |
| loss | 0.45 * BCEWithLogits + 0.55 * soft-Dice |

## Compute

Trained on one NVIDIA T4; the five folds needed 1.1 GPU-hours in total.

## Reproduce training

    python -m src.data_prep      --data-root data/MAGFiLO_1.0_Kaggle_2026 --cache work/cache
    python -m src.splits         --cache work/cache --n-folds 5 --seed 42
    python -m src.train_yolo_seg data  --cache work/cache
    python -m src.train_yolo_seg train --fold 0 --imgsz 1024 --epochs 30
    python -m src.train_refiner  train --fold 0 --crop 256 --epochs 8
    python -m src.infer_modeling --split val --fold 0     # OOF PQ with the official metric

The final submitted model is the fold-0 YOLO11s-seg detector + seed-conditioned crop-refiner cascade stored in weights/ (public LB 0.34); the selected final submission (public 0.39) is a public U-Net ensemble reproduction documented in README.md.
