# RF17 random split diagnostic

This is a diagnostic comparison, not a replacement for temporal CV or final test.

For each hotel and temporal fold, take that fold's eligible training bookings, make five stratified random 80/20 splits (seeds 17, 42, 73, 101, 2026), and fit the saved RF17/F1 preprocessing and model configuration on the 80%. The **same fitted model** scores its random 20% holdout and the fold's later temporal validation window. Report the mean across seeds. The original final test is never loaded.

| Hotel | Fold | Train cancel rate | Future cancel rate | Train 80% AUC | Random holdout AUC | Future AUC |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 1 | 8.34% | 6.46% | 0.873 | 0.766 | 0.640 |
| H1 | 2 | 7.97% | 5.79% | 0.868 | 0.760 | 0.744 |
| H1 | 3 | 7.69% | 9.90% | 0.871 | 0.768 | 0.615 |
| H1 | 4 | 7.74% | 9.50% | 0.866 | 0.768 | 0.643 |
| H2 | 1 | 15.35% | 8.30% | 0.757 | 0.728 | 0.619 |
| H2 | 2 | 14.01% | 13.49% | 0.740 | 0.716 | 0.635 |
| H2 | 3 | 13.74% | 13.22% | 0.734 | 0.722 | 0.603 |
| H2 | 4 | 13.62% | 11.62% | 0.725 | 0.708 | 0.593 |

Equal-weight mean AUC across folds: H1 **0.869 train / 0.765 random / 0.661 future**; H2 **0.739 / 0.719 / 0.612**. This shows both within-cohort generalization loss and an additional future-window loss for H1. For H2, the within-cohort loss is much smaller than the future-window loss.

H2's eligible training cancellation rate exceeds the future validation rate in all four folds. Fold 1 differs by 7.05 percentage points. Its RF17 mean predicted rate on future validation is about 15.5%, close to the 15.35% eligible training rate and well above the 8.30% observed rate. The base-rate shift explains much of this fold's average overprediction. H2 folds 2–3 have near-matching train/future cancellation rates yet still show lower future AUC than random-holdout AUC, so a prevalence change alone does not explain the ranking deterioration.

As a sensitivity check, H2 fold 1 has 958 earlier training candidates with labels still unavailable at fit time. Their eventual cancellation rate is 5.32%. Including them with hindsight would lower the earlier cohort's rate from 15.35% to 14.72%, still well above the future 8.30%. This does not make those rows valid for training at that fit date; it only shows that the label-availability filter explains part, not most, of the observed fold-1 rate difference.

**Limits:** The random holdout draws from rows with labels already available at each fit date, while future validation contains the full future cohort. Thus the AUC gap combines chronological change, label-availability selection, and other cohort differences. It is not a causal decomposition of overfitting versus drift. The diagnostic ran under Python 3.13/scikit-learn 1.7.2, while the original experiments used Python 3.12/scikit-learn 1.6.1. Compare results within this diagnostic; do not replace the frozen original CV metrics with them.

See `run_config.json` for settings, `seed_metrics.csv` for all five runs, `summary.csv` for per-fold means, and `../../diagnostics/random_split_diagnostic.py` for code.
