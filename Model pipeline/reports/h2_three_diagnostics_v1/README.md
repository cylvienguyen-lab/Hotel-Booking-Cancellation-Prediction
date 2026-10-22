# H2: three development diagnostics

Run: `C:/Users/Thao Vy/anaconda3/python.exe "Project/Model pipeline/diagnostics/h2_three_diagnostics.py"`

The cancellation experiments use the four existing H2 temporal folds and the same tuned RF settings as `feature_ablation_round1_v1`. Metrics below are unweighted means across folds. The final test was not used for model fitting or metrics. The separate label audit covers all H2 Gold rows, including the final-test period; it is a data integrity check, not an estimate of model performance.

## 1. Extra feature groups

| Feature set | Columns | ROC AUC | Average precision | Log loss | Brier |
|---|---:|---:|---:|---:|---:|
| F1 baseline | 17 | 0.6127 | 0.1781 | 0.3575 | 0.1025 |
| + booking state | 22 | **0.6610** | **0.2495** | **0.3434** | **0.0983** |
| + customer history | 19 | 0.6197 | 0.1715 | 0.3557 | 0.1021 |
| + external trends | 19 | 0.6090 | 0.1735 | 0.3577 | 0.1026 |
| + all nine Others | 26 | 0.6515 | 0.2376 | 0.3443 | 0.0987 |

The five booking-state columns are `BookingChanges`, `DaysInWaitingList`, `ADR`, `TotalOfSpecialRequests`, and `DepositType`. Their AUC gain over F1 was +0.0480, +0.0292, +0.0284, and +0.0875 in folds 1–4. Adding all nine Others performed slightly worse than the booking-state group alone. No set reached 0.75 AUC.

This is an exploratory feature-availability test, not a deployable model or a formal upper bound. The source booking table contains final-record values; it does not establish that these five values were known at T−14. They must be reconstructed from timestamped snapshots or events before being used in a real cutoff model. A low score here cannot prove that the label has a 0.65 ceiling.

## 2. Label audit

Across 38,233 H2 Gold bookings, positive labels are 3,931 `Canceled` and 605 `No-Show`; 33,697 `Check-Out` are negative. Among positives, the final status date is 1–13 days before arrival for 3,759 bookings and on arrival for 777. None has a final status date on or before T−14. In the Silver source scope, cancellations 13 days before arrival are eligible (235), whereas cancellations 14 and 15 days before arrival are excluded (331 and 400). All 26,261 early terminal bookings in scope are absent from Gold.

These checks support the implemented T−14 boundary and show that the label combines cancellation with no-show. They cannot verify the real cancellation event time because the source has a final `ReservationStatusDate`, not an event history.

## 3. Train–future adversarial validation

A separate RF predicts **which cohort a row came from**, using only the 17 F1 predictors. AUC 0.5 would mean the cohorts are hard to distinguish. Mean AUC over three random holdout seeds:

| Fold | Eligible train vs future | All earlier Gold candidates vs future |
|---|---:|---:|
| 1 | 0.855 | 0.838 |
| 2 | 0.765 | 0.752 |
| 3 | 0.815 | 0.799 |
| 4 | 0.826 | 0.825 |

For fold 1, removing the three calendar predictors still gives 0.845 (eligible) and 0.828 (all earlier candidates). The largest univariate differences in fold 1 without calendar fields are `LeadTime` (KS 0.307), `CustomerType` (total variation 0.276), and `MarketSegment` (0.266). Median `LeadTime` is 65 days in eligible train and 126 in future validation. `Transient` rises from 46.7% to 74.3%; `Online TA` rises from 40.9% to 64.4%. The large cohort difference persists when earlier Gold candidates replace fit-eligible train rows, though that comparison still uses Gold's T−14 eligibility and cannot isolate every selection effect.

The adversarial result establishes covariate shift in the observed predictors. It does not establish that this shift alone caused cancellation AUC or calibration loss.

## Files

- `upper_bound_fold_metrics.csv`: feature comparison for each fold.
- `upper_bound_summary.csv`: mean of the four folds.
- `label_audit.json`: label and T−14 boundary counts.
- `adversarial_seed_metrics.csv`: each random holdout run.
- `adversarial_summary.csv`: mean and SD over seeds.
- `feature_shift.csv`: per-feature univariate distance for each cohort comparison.
- `run_config.json`: features, parameters, and environment.

Environment: Python 3.13.9, scikit-learn 1.7.2. Original model pipeline was run under Python 3.12 and scikit-learn 1.6.1; F1 baseline metrics reproduced to the displayed precision.
