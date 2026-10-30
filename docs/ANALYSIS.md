# Quantitative analysis for the evaluation rubric

The organisers' rubric (Evaluation page) scores, under Quantitative Comparison (70%):
Panoptic Quality, the distribution of Dice scores, the distribution of IoU scores, and the distribution
of one-to-many / many-to-one relations between ground truth and predictions. This file records those
quantities for the final submission (public U-Net ensemble, reproduced), computed on grouped five-fold
out-of-fold predictions of the same pipeline (142 physical images, 232 annotator-image records).

## Protocol

For every annotator record we compute the pairwise IoU matrix between its masks and the predictions of the
same physical image; a pair with IoU > 0.5 is a true positive, predictions without a hit are FP and
annotations without a hit are FN. Dice is computed per matched pair as 2|A n B| / (|A| + |B|).

## Results (fold-0 out-of-fold)

| Quantity | Value |
|---|---|
| TP / FP / FN | 1238 / 528 / 579 |
| PQ / SQ / RQ | 0.4755 / 0.6881 / 0.6910 |
| Dice mean (std) | 0.812 (0.063) |
| Dice p05 / median / p95 | 0.701 / 0.818 / 0.903 |
| IoU mean (std) | 0.688 (0.089) |
| IoU p05 / median / p95 | 0.539 / 0.692 / 0.824 |
| IoU min / max (matched pairs) | 0.502 / 0.885 |
| One-to-many (one annotation matched by >=2 predictions) | 0 |
| Many-to-one (one prediction matched by >=2 annotations) | 1 (max 2 annotations) |

Notes: predictions are pixel-disjoint by construction, so a ground-truth object split by the detector
produces two false positives and one false negative rather than a one-to-many match; the only many-to-one
case is a predicted mask that spans two overlapping annotations.

## Reproduce

    python src/rubric_metrics.py   # writes work/analysis/ensemble/rubric_metrics_fold0.json

Inputs: `work/dariush_oof/pred_dariush_fold0.csv` (fold-aware out-of-fold predictions of the reproduced
public ensemble) and the official training annotations.
