"""Solar filament segmentation - reusable implementation modules.

At the code-freeze step, copy the finalised modules of the internal project here:

    data_prep.py          - COCO parsing, pycocotools rasterisation, instance caches
    splits.py             - grouped 5-fold split by physical JPEG
    pq_metric.py          - official Panoptic-Quality replica (per annotator pass)
    train_yolo_seg.py     - YOLO11-seg detector training (25.7M params, 1024 px)
    train_refiner.py      - seed-conditioned ResNet18 U-Net crop refiner
    infer_modeling.py     - inference, TTA, pixel-disjoint assembly, submission, PQ
    validate_submission.py- CSV schema / coverage / per-image overlap validation

Keep the module names and public functions stable: notebooks/final_pipeline.ipynb
imports them exactly as documented in src/README.md.
"""
