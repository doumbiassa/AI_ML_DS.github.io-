"""Skeleton Airflow DAG for the GridWise pipeline. Fill in the TODOs."""
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


def extract(ds, **_):
    # TODO: read incoming/readings_{ds}.csv (and late/ if present); detect delimiter
    ...


def validate(ds, **_):
    # TODO: apply validation_rules.yaml; send failures to a quarantine table
    ...


def load(ds, **_):
    # TODO: upsert into fact_reading (unique meter_id + reading_time), refresh agg_daily_usage
    ...


with DAG("gridwise_daily", start_date=datetime(2025, 6, 1), end_date=datetime(2025, 6, 30),
         schedule="@daily", catchup=True, tags=["teaching"]) as dag:
    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="validate", python_callable=validate)
    t3 = PythonOperator(task_id="load", python_callable=load)
    t1 >> t2 >> t3
