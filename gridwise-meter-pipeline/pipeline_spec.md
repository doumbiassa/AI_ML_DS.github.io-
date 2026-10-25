# Pipeline specification

**Goal:** load every daily file into a small warehouse (SQLite is fine) and publish daily usage per meter.

1. **Extract**: read every file in `incoming/` and `late/`. Detect the delimiter and decimal separator.
2. **Validate**: apply the rules in `validation_rules.yaml`. Rows that fail go to a `quarantine` table with the rule ID.
3. **Transform**: standardize column names (`kwh` → `energy_kwh`), parse timestamps, add `date`, join `reference/meters.csv`.
4. **Load**: `dim_meter`, `fact_reading` (unique on meter_id + reading_time) and `agg_daily_usage` (meter, date, total_kwh, peak_kwh, readings).
5. **Idempotency**: running the pipeline twice must not change the results.
6. **Orchestrate**: wrap the steps in an Airflow DAG (see `dags/gridwise_daily.py`) that runs once per day.

Check your `agg_daily_usage` against `reference/daily_usage_clean.csv` (computed before any problems were injected,
so small differences from dropped and quarantined readings are expected; explain them).
