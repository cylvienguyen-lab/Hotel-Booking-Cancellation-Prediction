# Hotel cancellation model development

`model_development.ipynb` develops cancellation or no-show risk models for H1
and H2 at 14 days before arrival (T-14). It uses the 30 predictors in the Gold
feature contract, four fixed temporal validation folds, and a separate Test
partition. The notebook compares models and saved out-of-fold predictions; it
does not fit a final model or evaluate Test outcomes.

## What a fresh clone needs

- Docker with the Compose plugin and enough disk space for the Python image and
  model dependencies.
- This repository with the current notebook, `development_config.json`,
  `Data Split Specification v2.docx`, `utils/`, `requirements.txt`, Dockerfile,
  and Compose file committed.
- The Gold data below, supplied separately. `datamart/` is ignored by Git, so a
  clone or pull alone will not contain it.

Place Gold under the sibling Data pipeline directory:

```text
Project/
├── Data pipeline/hotel_booking_pipeline/datamart/gold/
│   ├── feature_contract.json
│   ├── model_dataset/
│   │   ├── gold_model_dataset_H1.parquet/  (Parquet directory)
│   │   └── gold_model_dataset_H2.parquet/  (Parquet directory)
│   └── labels/
│       ├── gold_label_store_H1.parquet/    (Parquet directory)
│       └── gold_label_store_H2.parquet/    (Parquet directory)
└── Model pipeline/
    ├── model_development.ipynb
    ├── development_config.json
    ├── docker-compose.yaml
    └── utils/
```

Keep each `.parquet/` directory intact, including its part files. The notebook
reads Development predictors and labels from these paths and only booking
metadata for Test. To rebuild Gold instead, obtain the raw and external source
files separately and follow
[`../Data pipeline/hotel_booking_pipeline/README.ipynb`](../Data%20pipeline/hotel_booking_pipeline/README.ipynb).
Those source files are also ignored by Git.

Saved `splits/` and `experiments/` artifacts are optional for a fresh run. If
they are supplied, copy **complete** directories into `Model pipeline/` before
opening the notebook. Do not copy only `config.json`: the notebook rejects a
split or experiment directory with missing Parquet files. Both directories
are ignored by Git.

## Run with Docker

From `Project/Model pipeline`:

```sh
docker compose up --build -d
docker compose logs model-jupyter
```

Open the JupyterLab URL printed in the logs, on `127.0.0.1:8889`. Its token is
also printed there. Open `model_development.ipynb` and run all cells in order.
The first run creates missing split and experiment artifacts; later runs load
and verify matching artifacts. Sections with TabM and the other screening runs
can take substantially longer than the initial models on CPU.

When finished:

```sh
docker compose down
```

Compose mounts `Model pipeline/` read-write at
`/workspace/Project/Model pipeline` and the sibling `Data pipeline/` read-only
at `/workspace/Project/Data pipeline`. The notebook's default Gold path matches
these mounts. For a local Python run outside Docker, install
`requirements.txt`, start Jupyter from `Model pipeline/`, and set
`DATA_PIPELINE_ROOT` only if the Data pipeline is in a different location.

## Pipeline and saved files

`development_config.json` fixes the split calendar, experiment version, binary
predictor roles, and initial model parameters. `Data Split Specification v2.docx`
describes the approved split. Each training fold uses Development
bookings that meet all three conditions strictly before its fit date:

```text
CutoffDate < FitDate
LabelAvailableDate < FitDate
ExpectedCheckoutDate = ArrivalDate + TotalNights < FitDate
```

The notebook runs in this order:

1. Load Gold, create or verify `split_v2`, and run the initial Dummy, Logistic
   Regression, Random Forest, and XGBoost experiments for each hotel.
2. Assess saved validation predictions and inspect temporal, probability,
   segment, daily, and generalization diagnostics.
3. Screen individual changes in training strategy, features, RF settings,
   model family, and LR settings; then evaluate predefined combined runs.
4. Compare saved out-of-fold ensembles and forward-only Sigmoid/Isotonic
   calibration experiments.
5. Review the uncalibrated finalists. This comparison records no final model
   choice, final refit, or Test score.

`splits/split_v2/` contains `config.json`, `membership.parquet`, and
`audit.json`. Each run under `experiments/` contains `config.json`,
`metrics.parquet`, and `predictions.parquet`. The current notebook has 8 initial
runs, 36 screening runs, 7 combined runs, 12 ensemble runs, and 18 calibration
runs. Existing artifacts are checked instead of overwritten. A changed split
rule, calendar, feature set, or model setup needs a new versioned run; do not
mix files from different runs.