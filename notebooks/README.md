# Demo notebooks

Kaggle-ready notebooks that reproduce our selected final submissions end-to-end
(preprocessing -> inference with the published weights -> postprocessing -> evaluation).
Third-party checkpoints are attached as public Kaggle datasets, so the committee does not have to
download anything manually; our own code and weights live in this repository.

| notebook | submission | public LB | inputs |
|---|---|---|---|
| demo_submission_A.ipynb | ref 57768486, public 10x U-Net ensemble with 8-view TTA | 0.39 | competition data + dataset dariushafshar/solar-filament-unet-weights |
| demo_submission_B.ipynb | ref 57782016, 15-model YOLO consensus + our model-vote filter | 0.39 | competition data + dataset notoverkil/solar-filament-weights |
| final_pipeline.ipynb | our own cascade (self-trained weights in weights/) | 0.34 | competition data only |

Notes
- Each notebook lists its expected row/image counts and scores, and ends with the pooled-PQ evaluation
  (src/eval_demo.py) that reproduces the numbers in docs/report.pdf.
- Training is optional: A and B use published checkpoints (our training code is included for transparency),
  while final_pipeline.ipynb trains from scratch if the shipped weights/ directory is not used.
- To run: open the notebook on Kaggle, attach the listed dataset and the competition, then Run All.
