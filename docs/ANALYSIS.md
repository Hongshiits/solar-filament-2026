# Quantitative analysis for the evaluation rubric

The organisers' rubric (Evaluation page) scores, under Quantitative Comparison (70%): Panoptic Quality,
the distribution of Dice scores, the distribution of IoU scores, and the distribution of one-to-many /
many-to-one relations between ground truth and predictions. All numbers below are produced with the
**organisers' self-evaluation notebook** (azimahmadzadeh/self-evaluation-notebook) applied to the
out-of-fold predictions of the final submission (public 10x U-Net ensemble, 142 physical images,
232 annotator-image records, 1,817 annotations, 1,765 predicted masks).

Reproduce:

    python src/official_selfeval.py   # writes work/analysis/ensemble/official_selfeval_metrics.json

Inputs: `work/dariush_oof/pred_dariush_fold0.csv` (fold-aware out-of-fold predictions) and the official
training annotations; the script reproduces `rles_to_layers`, `get_overlap_matrices`, `get_overlap_df`,
`get_pq_score` and `plot_m2n_counts` exactly as in the organisers' notebook.

## Panoptic Quality

| quantity | value |
|---|---|
| PQ | 0.4755 |
| TP / FP / FN | 1238 / 528 / 579 |
| SQ (mean IoU of TP) / RQ | 0.6881 / 0.6910 |

## Distributions over all pairs with non-zero overlap (n = 1,602)

| quantity | p05 | p25 | median | p75 | p95 | mean |
|---|---|---|---|---|---|---|
| IoU | 0.200 | 0.527 | 0.655 | 0.739 | 0.818 | 0.605 |
| Dice | 0.334 | 0.690 | 0.791 | 0.850 | 0.900 | 0.734 |

The zero-overlap pairs are excluded from these histograms by the official plotting helper; including them
the means are 0.057 (IoU) and 0.070 (Dice) over all 16,916 GT-prediction pairs.

## One-to-many and many-to-one relations (overlap > 0)

| degree | annotations (1:n) | predictions (n:1) |
|---|---|---|
| 0 (no overlap) | 305 | 209 |
| 1 | 1428 | 1512 |
| 2 | 78 | 42 |
| 3 | 6 | 2 |

i.e. 84 of 1,817 annotations overlap two or three predictions, and 44 of 1,765 predictions overlap two
or three annotations. Because predictions are pixel-disjoint, split objects appear as two false positives
and one false negative under the IoU > 0.5 scoring rule rather than as a matched one-to-many pair.
