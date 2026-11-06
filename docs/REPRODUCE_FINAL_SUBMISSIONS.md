# Reproducing the selected final submissions

The organisers evaluate only what they can reproduce (Final-Submission discussion, 2026-10-10).
This repository contains code and data to reproduce both submissions we recommend selecting,
plus our self-trained cascade. Our own weights are stored in `weights/` (as required);
the two submissions below use *public* third-party weights that the jury can download from Kaggle.

## Submission A - public 10x U-Net ensemble, 8-view TTA (public LB 0.39, Kaggle ref 57768486)

Pipeline: 10 U-Nets (5 x resnet34 + 5 x tu-tf_efficientnetv2_s, one per grouped fold), trained on the
official training split only; 8-view D4 test-time augmentation; probability averaging; KEEP rule
(t_lo 0.35, t_hi 0.99, min area 200 px, mean probability >= 0.85); per-image pixel-disjoint assembly.

1. Download the public weights dataset: https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights
   (it ships the 10 checkpoints, `solar_unet_receipt.json`, `folds.csv`, and a segmentation-models-pytorch wheel).
2. Attach it, together with the competition data, to a Kaggle GPU script kernel and run
   `scripts/infer_dariush_ensemble.py`. `CONFIG['max_models']=0` uses all 10 models; `CONFIG['tta']=8`.
3. Expected result: `submission.csv` with 1177 instances over 177 of the 180 test images, pixel-disjoint;
   public leaderboard PQ = 0.39 (ref 57768486). The out-of-fold PQ of the same pipeline is 0.4755
   (SQ 0.6881, RQ 0.6910) on 232 annotator-image records.

## Submission B - 15-model YOLO consensus with model-vote filtering (public LB 0.39, Kaggle ref 57782016)

Pipeline: 15 public YOLO11m/YOLO26m instance-segmentation checkpoints (3 model families x 5 folds)
trained on the official training split; 8-view D4 TTA; cross-model consensus (greedy IoU >= 0.5
clustering, score-weighted pixel vote >= 0.5, min area 150 px, max 150 detections); then each consensus
instance is kept only if at least 15 of the 15 models support it (IoU > 0.5 match).

1. Download the public weights dataset: https://www.kaggle.com/datasets/notoverkil/solar-filament-weights
   (YOLO checkpoints in `p1/p2/p3-*` folders; the same dataset also contains an unrelated published submission,
   which we do **not** use).
2. Run `experiments/code/cloud_notoverkil_ensemble.py` in a Kaggle GPU kernel with the attached dataset
   and the competition test images (its module docstring documents the sharded test / assemble commands).
   This produces the unfiltered consensus CSV (2328 instances, 180 images) - a copy of that intermediate
   output is provided in `assets/nk_v3_unfiltered_consensus.csv` for convenience.
3. Apply the vote filter we computed from the per-model caches:

       python scripts/vote_filter_nk.py \
           --unfiltered-csv assets/nk_v3_unfiltered_consensus.csv \
           --support-npz assets/nk_v3_consensus_support.npz \
           --threshold 15 --out submission_vote15.csv

   Expected output: 1259 instances over 177 images, pixel-disjoint. The per-image RLE sets are identical
   to the submitted file (verified: 177/177 images); only row identifiers differ.
4. Public leaderboard PQ = 0.39 (ref 57782016); the same consensus without vote filtering scores 0.38-0.39
   and has roughly twice the instance density.

## Submission C - our self-trained YOLO11s-seg + crop refiner cascade (public LB 0.34)

Fully self-contained: our weights are in `weights/best_det.pt` and `weights/best_ref.pt`,
see `notebooks/final_pipeline.ipynb` and `docs/INFERENCE.md`.

## Which two to select

We recommend selecting A and B: both score 0.39 on the public leaderboard, they are genuinely different
strategies (semantic U-Net ensemble vs. instance-detector consensus), and both are reproducible from the
public weights plus the code and data in this repository.
## Organiser requirement of 2026-11-05 (demo notebook entry point)

The organisers clarified (discussion topic 755907) that the repository must ship a **demo notebook** that
reproduces the predictions **without re-training**, that the model weights may live anywhere as long as the
notebook can load them **without any manual download**, and that preprocessing/postprocessing code must be
part of the notebook. Size matters too: they ask for the final model file, not large training checkpoints.

How this repository satisfies it:
- `notebooks/demo_submission_A.ipynb` and `notebooks/demo_submission_B.ipynb` are the entry points; they attach
  the public Kaggle weight datasets by id (`dariushafshar/solar-filament-unet-weights`,
  `notoverkil/solar-filament-weights`) so the committee needs no download step;
- `notebooks/final_pipeline.ipynb` covers our own cascade, whose final weights (best_det.pt, best_ref.pt) are
  stored in `weights/` (~74 MB together, i.e. the final inference models rather than training checkpoints);
- preprocessing and postprocessing are in `src/`/`scripts/` and are called from the notebooks; training is
  optional and documented (`docs/TRAIN.md`);
- `src/eval_demo.py` reproduces the reported pooled PQ and the rubric distributions (verified: PQ 0.4755,
  SQ 0.6881, RQ 0.6910, TP/FP/FN = 1238/528/579 on the fold-0 out-of-fold predictions).
