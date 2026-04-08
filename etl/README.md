# Titanic ETL Pipeline

Simple ETL pipeline for Titanic dataset using Bruin for orchestration.

## Folder Structure

```
etl/
├── config/
│   └── s3_config.yaml       # S3 connection settings
├── tasks/
│   ├── __init__.py
│   ├── ingest.py            # Batch ingestion from source
│   ├── clean.py             # Data cleaning task
│   ├── transform.py         # Feature engineering
│   └── save.py              # Save to S3
├── dags/
│   └── titanic_dag.py       # Bruin DAG definition
├── data/
│   ├── raw/                 # Raw CSV files
│   └── processed/           # Training-ready datasets
├── .env.example             # Environment variables template
└── README.md                # This file
```

## Quick Start

### 1. Install dependencies

```bash
pip install pandas boto3 pyyaml bruin
```

### 2. Configure S3

```bash
cp etl/config/.env.example etl/config/.env
# Edit etl/config/.env with your S3 settings
```

### 3. Run locally

```bash
# Run individual tasks
python etl/tasks/ingest.py
python etl/tasks/clean.py
python etl/tasks/transform.py
python etl/tasks/save.py

# Or run the full DAG
python -m bruin etl/dags/titanic_dag.py
```

### 4. Run in cloud (AWS)

```bash
# Deploy to AWS Lambda via Bruin
bruin deploy etl/dags/titanic_dag.py --provider aws
```

## Tasks Overview

| Task      | Description                                            |
| --------- | ------------------------------------------------------ |
| ingest    | Downloads raw Titanic CSV from source to S3 raw bucket |
| clean     | Handles missing values, removes duplicates             |
| transform | Feature engineering (family_size, age bins, etc.)      |
| save      | Uploads processed data to S3 processed bucket          |

## S3 Buckets

- `titanic Raw data`: `titanic-etl-raw-{env}`
- `titanic Processed data`: `titanic-etl-processed-{env}`
