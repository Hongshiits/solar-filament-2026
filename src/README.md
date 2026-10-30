# Source modules

The notebook in ../notebooks imports the modules below. Module names and public entry points
are part of the reproduction contract: if you rename something, update the notebook in the same
commit.

| module | responsibility | key entry points |
|---|---|---|
| cloud_pipeline.py | **frozen final pipeline**: data discovery, grouped folds, YOLO11-seg detector training, seed-conditioned crop-refiner training, topology-safe flip-4 TTA, pixel-disjoint assembly, test inference and submission writing. The notebook imports its inference functions (load_detector, build_refiner_class, predict_image, entries_for_config, entry_to_full). | main(argv), load_detector(...), predict_image(...), entries_for_config(...) |
| train_yolo_seg.py | standalone YOLO11-seg single-class detector trainer (1024 px input, medoid/all-annotator targets) | data / train sub-commands |
| train_refiner.py | standalone seed-conditioned 256 px crop refiner (ResNet18 U-Net, BCE + soft-Dice) | train sub-command |
| infer_modeling.py | local inference for the trained cascade (detector -> refiner -> topology-safe TTA -> pixel-disjoint assembly -> COCO RLE), writes submission.csv | predict_split(...), write_submission(...) |
| data_prep.py | parse the COCO JSON, rasterise polygons with pycocotools (matches the official GT rasterisation), cache per-annotator instance label maps | main(...) |
| splits.py | grouped 5-fold split, unit = physical JPEG stem (an observation annotated by several people is never split across folds) | build_folds(...) |
| pq_metric.py | exact replica of the competition's Panoptic Quality: TP = every (GT, pred) pair with IoU > 0.5, FP/FN unmatched, pooled over annotator passes | evaluate(...) |
| validate_submission.py | schema, RLE decodability, positive area, unique filament_id, image coverage and per-image pixel-overlap check | main(csv_path) |
| ensemble.py | five-fold / multi-model prediction ensembling utilities (mask-IoU clustering + vote) used for the optional ensemble variants | main(...) |

Every training script writes a JSON config fingerprint (seed + thresholds) so the report,
configs/ and the produced submission.csv can be matched.
