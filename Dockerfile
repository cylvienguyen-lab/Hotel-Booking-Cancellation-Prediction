FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SPARK_LOCAL_HOSTNAME=localhost \
    JAVA_HOME=/usr/lib/jvm/default-java

RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jdk-headless procps libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Project
COPY requirements.txt /tmp/project-requirements.txt
RUN python -m pip install -r /tmp/project-requirements.txt

# Airflow runs in its own environment so its pinned dependencies cannot alter
# the tested ML stack used by the five pipeline jobs.
RUN python -m venv /opt/airflow-venv \
    && /opt/airflow-venv/bin/pip install --no-cache-dir \
      "apache-airflow==3.3.2" \
      "apache-airflow-providers-standard==1.19.0" \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.3.2/constraints-3.12.txt" \
    && /opt/airflow-venv/bin/pip check

COPY . /workspace/Project/

CMD ["python", "pipelines/data/run_data.py"]
