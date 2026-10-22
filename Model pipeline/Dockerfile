FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SPARK_LOCAL_HOSTNAME=localhost

# Spark needs Java and procps; XGBoost uses the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jdk-headless procps libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/model-requirements.txt
RUN python -m pip install -r /tmp/model-requirements.txt

EXPOSE 8889

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8889", "--no-browser", "--allow-root", "--ServerApp.root_dir=/workspace/Project/Model pipeline"]
