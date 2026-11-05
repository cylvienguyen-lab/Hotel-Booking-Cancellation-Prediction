"""Replay fixed hotel-booking inputs through historical Test evaluation."""

from __future__ import annotations

from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator


PROJECT_ROOT = "/workspace/Project"
PYTHON = "/usr/local/bin/python"


def job(task_id: str, script: str, cwd: str) -> BashOperator:
    log_dir = (
        f"{PROJECT_ROOT}/artifacts/airflow/logs/dag_id=bootstrap_demo/"
        f"run_id={{{{ run_id }}}}/task_id={task_id}"
    )
    return BashOperator(
        task_id=task_id,
        bash_command=(
            f"set -o pipefail; mkdir -p '{log_dir}' && "
            f"{PYTHON} {PROJECT_ROOT}/{script} 2>&1 "
            f"| tee '{log_dir}/attempt={{{{ ti.try_number }}}}.log'"
        ),
        cwd=f"{PROJECT_ROOT}/{cwd}",
        append_env=True,
    )


with DAG(
    dag_id="bootstrap_demo",
    description="Fixed historical Data to Test evaluation bootstrap",
    start_date=datetime(2017, 4, 18),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["hotel-booking", "bootstrap"],
) as dag:
    prepare_run = job("prepare_run", "pipelines/prepare_bootstrap.py", "pipelines")
    data = job("data", "pipelines/data/run_data.py", "pipelines/data")
    development = job(
        "development", "pipelines/model/run_development.py", "pipelines/model"
    )
    selection = job(
        "selection", "pipelines/evaluation/run_validation_selection.py", "pipelines/evaluation"
    )
    train = job("train", "pipelines/model/run_train.py", "pipelines/model")
    test_evaluation = job(
        "test_evaluation", "pipelines/evaluation/run_test_evaluation.py", "pipelines/evaluation"
    )

    prepare_run >> data >> development >> selection >> train >> test_evaluation
