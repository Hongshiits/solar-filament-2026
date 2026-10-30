# Inference contract (reproducing the submitted predictions)

The submitted predictions are produced **without training** by:

1. loading weights/detector.pt and weights/refiner.pt (stored in this repository),
2. running notebooks/final_pipeline.ipynb on the 180 official test images,
3. writing submission.csv and validating it.

## Inputs

| input | path (relative to repo root) |
|---|---|
| official competition data (unzipped) | data/MAGFiLO_1.0_Kaggle_2026/ |
| test images (H-alpha, 2048x2048 JPEG) | data/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg |
| detector weights | weights/detector.pt |
| refiner weights | weights/refiner.pt |
| frozen inference config | configs/inference.yaml |

No ground truth, no external data and no temporal neighbours are used at inference; only the single test image is passed to the models (competition rules section 6).

## Outputs

- submission.csv: two columns, filament_id and segmentation_rle; one row per predicted filament; compressed COCO RLE counts on a fixed 2048x2048 canvas.
- The local validation cell checks: schema, unique ids, RLE decodability, positive area, image coverage, and **per-image pixel overlap** (overlapping predictions make the official submission ERROR).

## Cost

| item | value |
|---|---|
| hardware | one NVIDIA T4 (15 GB) |
| end-to-end time for 180 images | 2 min (0.5 s/image) |
| peak GPU memory | not measured; fits comfortably in one T4 (15 GB) |
| inference config | detector imgsz 1024, refiner crop 256, conf 0.35, refiner threshold 0.50 |

## Determinism

- fixed random seed 42; inference itself is deterministic apart from GPU kernel non-determinism.
- no training or fine-tuning happens in the notebook.
- configs/inference.yaml must match the configuration used for the reported predictions; the report cites the same values.
