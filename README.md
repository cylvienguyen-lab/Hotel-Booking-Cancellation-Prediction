# Hotel booking bootstrap pipeline

This directory contains the fixed input snapshots and executable source for a
reproducible historical Data → Development → Validation selection → Train →
Test evaluation run. Generated data and model artifacts are ignored by Git.
Inference and monitoring are not part of this bootstrap yet.

## Requirements

- Docker Desktop running Linux containers and Docker Compose.
- Internet access for the first image build. The shared image installs CPU-only
  PyTorch; no GPU is required for this run.
- Run the commands below from the repository root.

## Layout

| Path | Purpose |
| --- | --- |
| `config/` | Source feature contract and Development configuration |
| `data/raw/`, `data/external/` | Fixed input snapshots |
| `data/datamart/` | Generated Bronze, Silver, and Gold tables |
| `pipelines/data/` | Data job and transformation functions |
| `pipelines/model/` | Development and Train jobs and model functions |
| `pipelines/evaluation/` | Validation selection and Test evaluation jobs |
| `notebooks/` | Development and Evaluation notebook code used by the job runners |
| `artifacts/development/` | Generated split, experiments, and Development audit |
| `artifacts/evaluation/` | Generated validation choice, Test results, and evaluation audits |
| `artifacts/train/` | Generated saved models and Train audit |
| `dags/` | Airflow `bootstrap_demo` DAG |
| `artifacts/airflow/` | Generated Airflow metadata and task logs |

## Run the complete bootstrap

```sh
docker compose run --build --rm bootstrap-demo
```

The `--build` flag includes the pinned Airflow environment added to the
Dockerfile; cached layers are reused on later runs. The command initializes
Airflow's local metadata database and executes one on-demand `bootstrap_demo`
DAG run. It does not require the Airflow web UI or scheduler.

The DAG runs these tasks in order:

```text
prepare_run → data → development → selection → train → test_evaluation
```

| Task | Main input | Main output |
| --- | --- | --- |
| `prepare_run` | Previous generated files, if any | Removes previous generated outputs |
| `data` | `data/raw/`, `data/external/`, `config/feature_contract.json` | `data/datamart/gold/` and `data_build_audit.json` |
| `development` | Gold tables, `config/development_config.json` | `artifacts/development/splits/`, `experiments/`, Development audit |
| `selection` | Development experiment results | `artifacts/evaluation/finalist_choice.parquet`, selection audit |
| `train` | Gold tables, split, selected model configuration | `artifacts/train/models/`, Train audit |
| `test_evaluation` | Saved models, historical Test cohort | Predictions, metrics, forecast and shortlist backtests in `artifacts/evaluation/` |

`prepare_run` deletes only generated `data/datamart/`,
`artifacts/development/`, `artifacts/evaluation/`, and `artifacts/train/` from a
previous run. Fixed inputs under `data/raw/` and `data/external/`, source code,
config, and Airflow metadata remain. This makes a second bootstrap run start
with fresh output rather than collide with immutable experiment artifacts.

The final success line is `Bootstrap demo passed: 5 jobs, 2 saved models,
11,324 historical Test predictions`. If a task fails, later tasks and the final
verification do not run. Each task also writes its console output under
`artifacts/airflow/logs/` so the UI can display it after the one-off bootstrap
container exits.

## Inspect the run in Airflow UI

After running the bootstrap command, start the optional local UI:

```sh
docker compose --profile ui up -d --build airflow-ui
```

Open `http://localhost:8080`, log in as `admin`, and read the generated
password from `artifacts/airflow/simple_auth_manager_passwords.json.generated`
on the host. In PowerShell, from the repository root:

```powershell
(Get-Content -Raw .\artifacts\airflow\simple_auth_manager_passwords.json.generated | ConvertFrom-Json).admin
```

The UI uses the same Airflow metadata database as the one-command
bootstrap, so open `bootstrap_demo` and its latest run to inspect the six task
states. The UI service stays running until stopped:

```sh
docker compose --profile ui stop airflow-ui
```

## Run jobs individually for debugging

From this directory, run the jobs in order:

```sh
docker compose run --rm data-job
docker compose run --rm development-job
docker compose run --rm selection-job
docker compose run --rm train-job
docker compose run --rm test-evaluation-job
```

There is one Dockerfile and one Compose file for the five ML jobs and Airflow.
Airflow is installed in a separate Python environment within the image, so
its dependencies do not change the pinned ML libraries. Individual job
commands assume a clean generated-output state. The Data job writes Gold
tables and
`data/datamart/gold/data_build_audit.json`. Development writes split and OOF
experiment artifacts and `artifacts/development/development_job_audit.json`.
Validation selection writes `artifacts/evaluation/finalist_choice.parquet` and
`selection_job_audit.json`. Train writes complete H1 and H2 model bundles under
`artifacts/train/models/` and `train_job_audit.json`. Test evaluation writes
historical Test predictions, metrics, forecast and shortlist backtest tables
under `artifacts/evaluation/`, followed by `test_evaluation_job_audit.json`.

Train uses Development rows only. The Test evaluation job scores the historical
Test cohort with the saved models. The operational shortlist excludes `y_true`;
it is a historical replay, not live inference.

To inspect the source notebooks in Jupyter:

```sh
docker compose --profile dev up model-jupyter
```
