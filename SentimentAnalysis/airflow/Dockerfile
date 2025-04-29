FROM apache/airflow:2.9.1

USER root

RUN apt-get update && \
    apt-get install -y gcc g++ libpq-dev && \
    apt-get clean

USER airflow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dags/ /opt/airflow/dags/
COPY modules/ /opt/airflow/modules/
COPY config/ /opt/airflow/config/
COPY plugins/ /opt/airflow/plugins/

ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

# Entrypoint and command are inherited from the base image
