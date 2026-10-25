# Messy Data Library

**Practice datasets for learning to clean, scale, visualize, wrangle and engineer data.**

Real data is rarely tidy. Names are spelled five different ways, dates come in three formats, files arrive late,
and some values are simply impossible. The Messy Data Library gives students realistic, synthetic datasets with
deliberately planted problems to find and fix, plus multi-file projects for data wrangling and data engineering.

**Website:** https://doumbiassa.github.io/AI_DS_ML.github.io/

Curated by **Moussa Doumbia, Ph.D.**, Expert in AI, Machine Learning and Data Science
Department of Mathematics, Howard University, Washington, DC

---

## What's inside

### 1. Practice datasets (27 datasets)

For data cleaning, scaling and visualization. Each dataset comes with:

- a **raw file** with planted data-quality problems (the one you work on)
- a **cleaned reference file** made from the same records, to check your results
- a **data dictionary**, a hidden list of the planted problems, and suggested exercises

| Record | Dataset | Subject | Level | Files |
|---|---|---|---|---|
| MDL-001 | Student Success & Study Habits | Education | Beginner | [raw](data/student-success.csv) · [clean](data/student-success_clean.csv) |
| MDL-002 | Riverton 311 Service Requests | Government | Intermediate | [raw](data/riverton-311.csv) · [clean](data/riverton-311_clean.csv) |
| MDL-003 | Harbor City Short-Term Rentals | Business | Intermediate | [raw](data/harbor-rentals.csv) · [clean](data/harbor-rentals_clean.csv) |
| MDL-004 | Bean There Coffee Sales | Business | Beginner | [raw](data/bean-there-coffee.csv) · [clean](data/bean-there-coffee_clean.csv) |
| MDL-005 | Community Clinic Vitals | Health | Intermediate | [raw](data/clinic-vitals.csv) · [clean](data/clinic-vitals_clean.csv) |
| MDL-006 | Music Track Audio Features | Entertainment | Beginner | [raw](data/track-features.csv) · [clean](data/track-features_clean.csv) |
| MDL-007 | Craft Beer Chemistry | Food & Drink | Beginner | [raw](data/craft-brew.csv) · [clean](data/craft-brew_clean.csv) |
| MDL-008 | Maple County Home Sales | Real Estate | Intermediate | [raw](data/maple-county-housing.csv) · [clean](data/maple-county-housing_clean.csv) |
| MDL-009 | AirWatch Sensor Readings | Environment | Advanced | [raw](data/airwatch-sensors.csv) · [clean](data/airwatch-sensors_clean.csv) |
| MDL-010 | LinkUp Telecom Customer Churn | Business | Intermediate | [raw](data/linkup-churn.csv) · [clean](data/linkup-churn_clean.csv) |
| MDL-011 | ShopNest Online Orders | Business | Intermediate | [raw](data/shopnest-orders.csv) · [clean](data/shopnest-orders_clean.csv) |
| MDL-012 | Glide Bike-Share Trips | Transportation | Beginner | [raw](data/glide-bikeshare.csv) · [clean](data/glide-bikeshare_clean.csv) |
| MDL-013 | Fictional Nations Indicators | Social Science | Beginner | [raw](data/nations-indicators.csv) · [clean](data/nations-indicators_clean.csv) |
| MDL-014 | Riverton Restaurant Inspections | Government | Intermediate | [raw](data/riverton-inspections.csv) · [clean](data/riverton-inspections_clean.csv) |
| MDL-015 | Summit Analytics Employee Records | Business | Intermediate | [raw](data/summit-hr.csv) · [clean](data/summit-hr_clean.csv) |
| MDL-016 | Metro Basketball League Stats | Sports | Beginner | [raw](data/metro-hoops.csv) · [clean](data/metro-hoops_clean.csv) |
| MDL-017 | Kalindra Malaria Surveillance | Health | Intermediate | [raw](data/kalindra-malaria.csv) · [clean](data/kalindra-malaria_clean.csv) |
| MDL-018 | Tuberculosis Treatment Outcomes | Health | Intermediate | [raw](data/tb-outcomes.csv) · [clean](data/tb-outcomes_clean.csv) |
| MDL-019 | Memory & Aging Cohort | Health | Advanced | [raw](data/memory-cohort.csv) · [clean](data/memory-cohort_clean.csv) |
| MDL-020 | Heart Disease Risk Study | Health | Beginner | [raw](data/heart-disease.csv) · [clean](data/heart-disease_clean.csv) |
| MDL-021 | COVID-19 County Daily Reports | Health | Intermediate | [raw](data/covid19-counties.csv) · [clean](data/covid19-counties_clean.csv) |
| MDL-022 | Influenza-Like Illness Surveillance | Health | Intermediate | [raw](data/flu-surveillance.csv) · [clean](data/flu-surveillance_clean.csv) |
| MDL-023 | Common Cold Symptom Diary | Health | Beginner | [raw](data/common-cold-diary.csv) · [clean](data/common-cold-diary_clean.csv) |
| MDL-024 | Riverton Crime Incidents | Public Safety | Intermediate | [raw](data/riverton-crime.csv) · [clean](data/riverton-crime_clean.csv) |
| MDL-025 | Gun Violence Incidents | Public Safety | Intermediate | [raw](data/gun-violence-incidents.csv) · [clean](data/gun-violence-incidents_clean.csv) |
| MDL-026 | Regional Flight Delays | Transportation | Intermediate | [raw](data/regional-flights.csv) · [clean](data/regional-flights_clean.csv) |
| MDL-027 | Fuel & Crude Oil Prices | Energy | Beginner | [raw](data/fuel-prices.csv) · [clean](data/fuel-prices_clean.csv) |

### 2. Data Wrangling & Engineering (8 bundles)

Multi-file, multi-format projects mapped to the weeks of the **Data Wrangling / Data Engineering** course:
APIs, web scraping, data integration, SQL, validation, ETL pipelines, big data and log parsing.
Each bundle has its own README with its files, structure, tasks by week, starter code and planted challenges.

| Bundle | Name | Weeks | Formats |
|---|---|---|---|
| DWE-01 | [Campus Events API](wrangling/campus-events-api/README.md) | 3, 4, 14 | JSON, REST-style API |
| DWE-02 | [Riverbend Books Web Store](wrangling/riverbend-books-web/README.md) | 3, 5 | HTML, Web scraping |
| DWE-03 | [Northstar Retail Integration](wrangling/northstar-retail-integration/README.md) | 4, 5, 6 | CSV, JSON, Excel, TSV |
| DWE-04 | [Riverton General Hospital Database](wrangling/riverton-hospital-db/README.md) | 7, 8, 9 | SQLite, SQL, CSV |
| DWE-05 | [GridWise Smart-Meter Pipeline](wrangling/gridwise-meter-pipeline/README.md) | 9, 10, 11, 12 | CSV, YAML, Airflow |
| DWE-06 | [Riverton Taxi Trips (Big Data)](wrangling/riverton-taxi-bigdata/README.md) | 5, 13 | Parquet, Partitioned data, CSV |
| DWE-07 | [Station Climate Exports](wrangling/station-climate-wide/README.md) | 2, 5, 6 | CSV (wide), Excel (multi-row header) |
| DWE-08 | [Learning Platform Web Logs](wrangling/webserver-logs/README.md) | 3, 5, 14 | Log files (gzip), JSON Lines, Regex |

---

## Quick start

Load any practice dataset straight from the website:

```python
import pandas as pd

df = pd.read_csv("https://doumbiassa.github.io/AI_DS_ML.github.io/data/student-success.csv")
df.head()
```

Call the Campus Events API like a real web API:

```python
import requests
import pandas as pd

resp = requests.get("https://doumbiassa.github.io/AI_DS_ML.github.io/wrangling/campus-events-api/api/events/page-1.json", timeout=30)
resp.raise_for_status()
events = pd.json_normalize(resp.json()["data"], sep="_")
```

## Repository structure

```
├── index.html      the website
├── DR_MD.png       curator photo
├── data/           27 practice datasets (raw and _clean versions)
├── wrangling/      8 data wrangling & engineering bundles (one folder and one .zip each)
├── source/         Python scripts that generated all the data
└── .nojekyll       tells GitHub Pages to serve the site as is
```

## For students

1. Pick a dataset or bundle that fits your project.
2. Work on the **raw** files: explore, list every problem you find, fix it and write down why.
3. Only then open the planted-problems list and compare your results with the reference files.

## How to cite

> Doumbia, M. (2026). *Messy Data Library: Synthetic practice datasets for data science education* [Data set]. Howard University. https://doumbiassa.github.io/AI_DS_ML.github.io/

Cite a single dataset by its record number, for example *Messy Data Library, record MDL-001*.

## Note

All people, places, organizations and numbers are invented. The data imitates real-world patterns but is intended
for teaching only and must not be used to draw conclusions about real people, places, diseases or markets.
