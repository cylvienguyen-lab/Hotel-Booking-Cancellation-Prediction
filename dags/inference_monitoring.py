"""Replay a cutoff window one day at a time, then assess saved predictions."""
from datetime import datetime

from airflow.sdk import DAG, Param
from airflow.providers.standard.operators.bash import BashOperator


def replay_task(task_id):
    return BashOperator(
        task_id=task_id,
        bash_command=f"/usr/local/bin/python /workspace/Project/pipelines/operations/{task_id}.py",
        env={
            "REPLAY_START": "{{ params.start }}",
            "REPLAY_END": "{{ params.end }}",
            "REPLAY_AS_OF": "{{ params.as_of }}",
            "REPLAY_RUN_ID": "{{ run_id }}",
        },
        append_env=True,
    )


with DAG(
    dag_id="inference_monitoring",
    start_date=datetime(2017, 4, 18),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    params={
        "start": Param("2017-05-01", type="string", format="date"),
        "end": Param("2017-07-31", type="string", format="date"),
        "as_of": Param("auto", type="string"),
    },
    tags=["hotel-booking", "historical-replay"],
) as dag:
    inference = replay_task("inference")
    monitoring = replay_task("monitoring")
    report = replay_task("report")

    inference >> monitoring >> report
