# Evaluation pipeline

The third of three pipelines in this project. Produces the two
business-facing batch outputs (hotel-level forecast + booking-level
risk shortlist) that operations teams would consume.

## Project overview — the three pipelines

```text
┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────────┐
│   Data pipeline       │ →  │   Model pipeline      │ →  │   Evaluation pipeline     │
│                       │    │                       │    │   (THIS FOLDER)           │
│ Bronze / Silver /     │    │ Experiment screening  │    │ Val-OOF comparison        │
│ Gold medallion build  │    │ (Dummy / LR / RF /    │    │ Finalist selection        │
│                       │    │  XGB / LightGBM /     │    │ Calibration selection     │
│ Raw hotel CSVs +      │    │  CatBoost / TabM)     │    │ Operating-point selection │
│ holidays, FX,         │    │ Ensembling +          │    │ Test-set evaluation       │
│ Google Trends         │    │ calibration           │    │ Two business deliverables │
│ ~119k bookings,       │    │ Refit on all Dev +    │    │ Feature importance        │
│ 2 hotels              │    │ model registry        │    │ Monitoring hooks          │
└───────────────────────┘    └───────────────────────┘    └───────────────────────────┘
   feature contract             experiments/ dir             outputs/ dir
   datamart/gold/               models/ registry             (reads from above two)
```

Each pipeline has its own folder, README and (where relevant) its own
notebook. The Docker container ships from `Model pipeline/` and mounts
all three folders — see "How to run" below.

## What the notebook delivers

`evaluation_notebook.ipynb` covers everything in one place:

**Validation OOF (Sections 2 – 8)**
- Loads the shortlisted candidate experiments per hotel plus their
  Sigmoid and Isotonic calibrated variants.
- Computes a single, task-aligned metric set per (candidate, calibration):
  **Daily nRMSE** (primary), **Daily Bias %**, **Log Loss**,
  **AP / AP lift**, **ECE** (calibration tiebreaker). Both pooled and
  per-fold with mean±std for temporal stability.
- Picks one finalist per hotel + one calibration method per hotel, with
  written justification tied to the committed business action.

**Test cohort (Sections 9 – 12)**
- Reruns the metric set on `outputs/test_predictions.parquet` (produced
  by `../Model pipeline/model_refit.ipynb`).
- **Deliverable 1 — Hotel-level forecast (Section 10)**: nightly
  expected arrivals (check-ins) and expected room-nights occupancy, per
  hotel, model vs constant-rate dummy. Includes an explicit coverage
  ratio showing what fraction of total nightly occupancy comes from the
  Test cohort (97-99% on average; the first 5 days of May have lower
  coverage due to Dev-partition stays extending in — see the markdown
  in Section 10).
- **Deliverable 2 — Booking-level risk shortlist (Section 11)**:
  workload-based top-K per cutoff day, with two rankings (`p_cancel`
  and `expected_lost_revenue`), K chosen on validation and reported on
  test, Precision@K / Recall@K / Lift@K with mean-daily-prevalence
  baseline. Ships as two files: `_eval.parquet` with y_true for
  backtesting, `_operational.parquet` without y_true but with
  `model_version`, `prediction_timestamp`, `risk_rank` for the
  downstream batch system.
- Business-action summary and future-work scoping (Section 12).

The notebook never touches the Test cohort for selection decisions —
Test numbers are only read once, at the chosen operating point and
chosen finalist, for reporting.

## What it needs on disk

```text
Project/
├── Data pipeline/hotel_booking_pipeline/datamart/gold/
│   ├── model_dataset/gold_model_dataset_H1.parquet/
│   ├── model_dataset/gold_model_dataset_H2.parquet/
│   └── labels/gold_label_store_H1.parquet/, gold_label_store_H2.parquet/
├── Model pipeline/
│   ├── splits/split_v2/            <- membership.parquet + config.json
│   ├── experiments/                <- one folder per saved run
│   │   ├── H1_E2_RF30_XGB30_e1/
│   │   ├── H1_RF_RW365_s1/
│   │   ├── H1_RF_v2/, H1_XGB_v2/       (standalone components for H1)
│   │   ├── H2_E4_*_e1/, H2_E3_*_e1/, H2_E1_*_e1/
│   │   ├── H2_C2_RW180_FE4_c1/, H2_C4_T5_RW90_c1/
│   │   ├── H2_TabM_s1/                 (TabM-E8 standalone)
│   │   └── <*_Uncalibrated_p1, *_Sigmoid_p1, *_Isotonic_p1 for each>
│   └── model_refit.ipynb               <- produces test_predictions.parquet
└── Evaluation pipeline/            <- this folder
    ├── evaluation_notebook.ipynb
    ├── utils/eval_io.py, eval_metrics.py
    └── outputs/
        ├── test_predictions.parquet           <- from model_refit.ipynb
        ├── hotel_forecast_*.parquet           <- Deliverable 1
        ├── booking_shortlist_*.parquet        <- Deliverable 2
        └── layer_a_*.parquet                  <- metric tables
```

`splits/`, `experiments/`, and `datamart/` are gitignored (see root
`.gitignore`). Copy the complete folders in from Cylvie's run or
regenerate them by running `../Model pipeline/model_development.ipynb`
and `../Data pipeline/hotel_booking_pipeline/README.ipynb` end to end.

## How to run

Use the same Docker image Section 3 uses. From `Project/Model pipeline`:

```sh
docker compose up --build -d
docker compose logs model-jupyter
```

Open the JupyterLab URL and token printed in the logs (default
`127.0.0.1:8889`). The compose file now mounts three paths:

- `./` (`Model pipeline/`) read-write
- `../Data pipeline/` read-only
- `../Evaluation pipeline/` read-write

Run the notebooks in this order:

1. `Model pipeline/model_development.ipynb` — produces
   `experiments/` and `splits/split_v2/`. Already done if you have
   Cylvie's run.
2. `Model pipeline/model_refit.ipynb` — fits the two finalists on all
   eligible Dev bookings and writes `test_predictions.parquet` to the
   Evaluation pipeline's outputs folder. Takes a few minutes for
   RF+XGB (H1) and ~10-20 minutes for C2+TabM on CPU (H2).
3. `Evaluation pipeline/evaluation_notebook.ipynb` — runs top to
   bottom in <30 seconds. Produces all the metric tables, the two
   business-action output files, and all the plots.

The notebook writes everything to `Evaluation pipeline/outputs/`.

## What the notebook does NOT do

- It does not refit any Section 3 model. Every val-OOF number comes
  from the saved `predictions.parquet` files; every test number comes
  from the refit's `test_predictions.parquet`.
- It does not run a direct overbooking simulation. The deliverables are
  the hotel-level forecast and the booking-level outreach shortlist;
  direct overbooking is scoped as future work in Section 12 because the
  dataset lacks a true capacity figure and a measured walk cost.
- It does not persist models or do any refit itself. That is
  `../Model pipeline/model_refit.ipynb`'s job; this notebook consumes
  that notebook's outputs (`test_predictions.parquet` and the
  `../Model pipeline/models/` registry).

## Finalists (committed)

| Hotel | Finalist model | Calibration |
|---|---|---|
| H1 | H1-E2 — RF30 + XGB30 equal-weight ensemble | Uncalibrated |
| H2 | H2-E1 — C2 + TabM-E8 equal-weight ensemble | Uncalibrated |

Rationale: see Section 5 (model-quality) and Section 6 (calibration) of
the notebook. H2's choice is a conscious tradeoff — TabM-E8 standalone
wins the Daily nRMSE primary by a small margin but H2-E1 wins AP (the
ranking metric the outreach shortlist consumes) by a clear 15% relative
margin, which matches the committed business action.

---

# Handoff for the deployment / MLOps team

This section is for the engineer picking up the model for production.

## sklearn version compatibility (important!)

The sklearn pipeline pickles are **tied to the sklearn version that
produced them**. The project pins **sklearn 1.6.1** in `Model
pipeline/requirements.txt` and the Dockerfile. Loading these pickles
in an environment with a different sklearn version typically fails
with errors like `Can't get attribute '_RemainderColsList'` because
sklearn's internal `ColumnTransformer` representation changes between
minor versions.

Rules for the handoff and for running Section 13 of the evaluation
notebook:

- The **inference / deployment environment must run sklearn 1.6.1**
  (same as training). This is a hard requirement.
- When you upgrade sklearn in production, you must re-fit the models
  on the same training cohort and re-publish the pickles. The refit
  flow (`Model pipeline/model_refit.ipynb`) is designed to be
  reproducible exactly for this purpose.
- To run the eval notebook's feature-importance section on your local
  machine, pin sklearn to 1.6.1 (`pip install scikit-learn==1.6.1`)
  or run that section inside the project Docker container.

The eval notebook's Section 13 handles the version-mismatch error
gracefully — it prints the sklearn version detected in the pickle vs
the one in the current environment and tells you what to do.

## What you receive

```text
Model pipeline/models/<model_version>/
├── metadata.json                       # the manifest — read this first
├── <component>_pipeline.pkl            # sklearn Pipeline (RF30 / XGB30 / C2)
├── <component>_fe4_mapping.json        # C2 only — segment → cancel rate
└── <component>_tabm_artifact.json      # TabM-E8 only — spec + preprocessing
```

Two model versions ship, one per hotel:

- `H1-E2_v1_<date>` — H1-E2 ensemble (RF30 + XGB30), two pickles only.
  Fully self-contained; sklearn is all you need.
- `H2-E1_v1_<date>` — H2-E1 ensemble (C2 + TabM-E8). C2 ships as a
  sklearn pipeline + FE4 JSON; **TabM-E8 ships as a torch state dict +
  preprocessing JSON + architecture JSON**, ready to score without a
  refit. See the component in metadata.json for the exact file names
  and the architecture to pass to `TabM.make(...)`.

`metadata.json` is the source of truth. It includes:

- `model_version`, `ensemble_id`, `hotel`, `fit_date`, `created_at_utc`.
- `library_versions` — pin these in the production environment.
- `split_v2_fingerprint` — a short hash of the training cohort key set,
  for traceability if the split is ever regenerated.
- `components` — one entry per model component with:
  - `artifact_type`: `sklearn_pipeline_pickle` or `tabm_spec_json`.
  - `artifact_path`: the file to load (relative to the `metadata.json`).
  - `sha256`: integrity check.
  - `feature_columns`: exactly the columns to pass to
    `pipeline.predict_proba(...)` for sklearn components.
  - `fe4_mapping_path` and `fe4_mapping_sha256` for C2 only.

## Scoring interface

`Model pipeline/utils/model_scoring.py` is the reference implementation
of loading + scoring. Import it from Python:

```python
from utils.model_scoring import load_hotel_model, score_bookings

# One-time per deploy — load the current registered model
h1_model = load_hotel_model("H1", models_root=Path("Model pipeline/models"))
h2_model = load_hotel_model("H2", models_root=Path("Model pipeline/models"))

# Daily batch at CutoffDate = today
scores_h1 = score_bookings(h1_model, bookings_df_h1)
scores_h2 = score_bookings(h2_model, bookings_df_h2)
```

The returned frame has `Hotel`, `Booking_ID`, `p_cancel`,
`model_version`, plus `p_<component>` columns for audit. Both hotels
work out of the box.

### Input columns required at scoring time

The sklearn pipelines consume the Gold feature contract columns listed
in `metadata.json → components[*].feature_columns`. For C2 you also
need `MarketSegment` so the `HistoricalSegmentCancelRate` can be
computed from the FE4 mapping; `model_scoring.py` handles that
automatically. For TabM-E8, the preprocessing bundle lists exact
numeric and categorical columns required; the scorer applies the same
median imputation + standard scaling + categorical mapping that
training used.

**You do not need the full Gold pipeline at scoring time.** You need
the Silver-level transformations plus the external joins (holidays, FX,
Google Trends) that produce the Gold predictor columns. The data
pipeline in `Data pipeline/hotel_booking_pipeline/` implements this.

### Runtime dependencies

Both hotels at inference time need Python with:
- `scikit-learn==1.6.1` (hard pin — see version-compatibility section above)
- `xgboost` (for H1-E2's XGB30 component)
- `torch==2.8.0` + `tabm==0.0.3` (for H2-E1's TabM-E8 component)
- `numpy`, `pandas`, `pyarrow`

All of these are in `Model pipeline/requirements.txt`. The simplest
production environment is a thin Docker image built from the same
Dockerfile.

## Suggested deployment pattern (batch inference)

Daily at 02:00 local time, for each hotel:

1. Read all bookings where `CutoffDate = today` and the booking is
   active (not yet cancelled). Join holidays / FX / Google Trends from
   the Silver scoring tables.
2. Score with `score_bookings(model, bookings_df, as_of_date=today)`.
3. Build the two deliverables with the standalone functions in
   `Model pipeline/utils/model_outputs.py`:
   - `aggregate_hotel_forecast(scored)` → `{"arrivals": ..., "occupancy": ...}`
   - `generate_booking_shortlist(scored, k_per_hotel={"H1":10,"H2":20}, rank_by="p_cancel")`
4. Write both to the ops dashboard. The eval notebook's Sections
   10–11 are the reference implementation and call the same two
   functions, so schemas are guaranteed to match.
5. Log every prediction row with its `model_version`,
   `prediction_as_of_date`, and `generated_at_utc` for monitoring.

## Scoring-output schema

Every row produced by `score_bookings` and the two aggregation
functions carries three audit fields the deploy team must preserve
through to the ops dashboard and monitoring store:

| Field | Type | What it means |
|---|---|---|
| `model_version` | string | The model registry folder name (e.g. `H1-E2_v1_20170418`). Joinable 1-to-1 to `Model pipeline/models/<model_version>/metadata.json`. |
| `prediction_as_of_date` | date (YYYY-MM-DD) | The *business cutoff* the prediction represents. In production this is the batch date (`today`). In the backtest it is the booking's CutoffDate. This is the field monitoring uses when it asks "how did the model do on predictions made on 2017-06-15?". |
| `generated_at_utc` | datetime (ISO) | The actual UTC wall-clock when the row was written. Useful for job-run lineage (which batch invocation produced this?); not meaningful for business time. |

Splitting these two fields matters because in a real-time scenario
you may re-score the same cohort twice on different days (e.g. retrain,
rerun), and monitoring needs both the business-time axis
(`prediction_as_of_date`) and the batch-run axis (`generated_at_utc`).

## Monitoring hooks (what to watch)

Short list, lifted from the Section 4 writeup:

- **Prevalence drift.** Mean actual cancellation rate per hotel per
  week. Compare to training-time 7-13%. The H2 test window already
  showed drift from 13% to 8%, so this is real.
- **Model bias.** `100 × Σ(p − y) / Σ(y)` by hotel on the previous
  2-week outcome window. Trip if |bias| > 10%.
- **Coverage ratio.** Short-lead booking share. If the Silver /
  (Silver − EligibleT14) ratio moves out of the 85-92% band seen on
  Test, the hotel-level forecast's "+ PMS short-lead" integration
  needs re-tuning.
- **Top-feature distribution drift.** The `feature_importance.parquet`
  in `Evaluation pipeline/outputs/` identifies which inputs matter.
  Monitor those by population statistics (mean, variance, missingness).
- **Shortlist precision.** For the operations shortlist, compute
  Precision@K monthly on the realised outcomes. Expected ~15% on H1
  @ K=10 and ~14% on H2 @ K=20; alert if it drops below 10%.

The monitoring design is the Section 6 piece of the final
presentation and is written up alongside this handoff.
