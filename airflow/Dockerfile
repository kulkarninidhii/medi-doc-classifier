FROM apache/airflow:2.9.2-python3.11

# System deps for building transitive deps (e.g., xmlsec)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential \
    libxml2-dev libxmlsec1-dev libxmlsec1-openssl pkg-config \
 && rm -rf /var/lib/apt/lists/*
USER airflow

# Use Airflow constraints so provider versions match Airflow core
ARG AIRFLOW_VERSION=2.9.2
ARG PYTHON_VERSION=3.11
ARG CONSTRAINTS_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

RUN pip install --no-cache-dir --constraint "${CONSTRAINTS_URL}" \
    apache-airflow-providers-amazon \
    apache-airflow-providers-postgres \
    apache-airflow-providers-docker \
    boto3 \
    "psycopg[binary]"
