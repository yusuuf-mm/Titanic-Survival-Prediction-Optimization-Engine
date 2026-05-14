# Titanic Survival Prediction & Lifeboat Optimization Engine

A premium, full-stack decision intelligence system that fuses **Machine Learning** predictions with **Operations Research** optimization. This project not only predicts passenger survival using an advanced ensemble model but also solves the critical resource allocation problem: "How to maximize survivors under strict lifeboat capacity and ethical constraints?"

![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Docker](https://img.shields.io/badge/Docker-enabled-blue)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20DynamoDB-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🌟 Key Features

- 🔮 **ML Prediction API**: High-performance survival classifier powered by **XGBoost**.
- 🧮 **Optimization Engine**: Mixed-Integer Programming (MIP) solver using **PuLP** to allocate limited lifeboat seats based on predictive risk, family cohesion, and ethical priority rules (children and women first).
- 🎨 **Executive Dashboard**: A stunning, modern **Streamlit** UI for real-time inference, visualization of demographic selection, and interactive 3D rescue maps.
- ☁️ **Cloud Native Architecture**: 
  - Centralized model storage using **AWS S3** and **MLflow Registry**.
  - Rate-limiting and prediction logging distributed across instances via **AWS DynamoDB**.
- 🐳 **One-Command Deployment**: Fully containerized API and Dashboard with **Docker Compose**, including intelligent auto-training capabilities on startup.
- 🧪 **Robust Testing & CI/CD**: Pytest suite and GitHub Actions workflow for continuous integration and automated deployment to AWS.

---

## 🏗️ Project Architecture

```
Titanic-Optimization-Engine/
│
├── api/ (Root)
│   ├── predict.py             # FastAPI prediction, optimization, rate-limiting & logging
│   ├── train.py               # XGBoost training pipeline integrated with MLflow
│   ├── entrypoint.sh          # Intelligent startup & auto-training script
│   └── tests/                 # Pytest suite
│
├── dashboard.py               # Streamlit Executive Dashboard (3D Maps, KPIs)
│
├── optimization/
│   ├── __init__.py            # Optimizer initialization
│   └── lifeboat_optimization.py # MIP Optimization core logic
│
├── data/
│   └── titanic.csv            # Historical passenger dataset
│
├── Dockerfile                 # API Container definition
├── Dockerfile.dashboard       # Dashboard Container definition
└── docker-compose.yml         # Full system orchestration
```

### System Design

- **Frontend**: A sleek, dark-themed Streamlit application that retrieves real-time predictions and visualizations from the FastAPI backend.
- **Backend API**: A FastAPI application that serves the ML model predictions and runs the optimization algorithms. It includes DynamoDB-backed distributed rate-limiting.
- **Model Registry & Tracking**: Uses MLflow for experiment tracking and artifact registry. Core machine learning artifacts (like tokenizers/scalers) fallback and upload cleanly to AWS S3.
- **Data Persistence**: Inference logs are piped directly into a DynamoDB table for auditing and continuous learning.
- **Automation**: Docker-compose ensures both the API and dashboard spin up in tandem. `entrypoint.sh` trains the model automatically if artifacts are missing on first start.

---

## 🚀 Quick Start (Recommended)

The entire system is vertically integrated. You do not need to manually train models or install local dependencies if you have Docker.

### 1. Set Environment Variables (Optional but recommended for AWS integration)
Create a `.env` file in the root directory:
```env
S3_BUCKET_NAME=your-s3-bucket
DYNAMODB_TABLE_NAME=predictions
RATE_LIMIT_TABLE=rate-limits
AWS_REGION=us-east-1
ENABLE_RATE_LIMIT=false  # Set to true to enable DynamoDB rate limiting
UPLOAD_TO_S3=false       # Set to true to store artifacts in AWS S3 automatically
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### 2. Start the Engine
```bash
docker compose up --build -d
```

### 3. Access the System
- **Executive Dashboard**: `http://localhost:8501`
- **Inference API**: `http://localhost:8000/docs`

---

## 💻 Manual Setup (Development)

### 1. Installation
```bash
git clone https://github.com/yusuuf-mm/Titanic-Survival-Prediction-Optimization-Engine.git
cd Titanic-Survival-Prediction-Optimization-Engine
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Training Pipeline
Train the base XGBoost model and store artifacts:
```bash
python train.py
```

### 3. Run Services Independently
```bash
# Terminal 1: Launch FastAPI Engine
uvicorn predict:app --reload --port 8000

# Terminal 2: Launch Streamlit Dashboard
streamlit run dashboard.py
```

### 4. Running the Tests
```bash
pytest tests/
```

---

## 🧮 The Optimization Problem

We treat lifeboat allocation as a **Mixed-Integer Programming (MIP)** problem to demonstrate Operations Research in critical scenarios.

**Objective:**
Maximize $\sum (p_i \cdot x_i)$ where $p_i$ is the predicted survival probability of passenger $i$, and $x_i$ is the binary decision to allocate a seat.

**Constraints Support:**
1. **Capacity Limits**: Total allocated seats cannot exceed `Available Lifeboat Capacity`. ($\sum x_i \leq \text{Available Seats}$)
2. **Vulnerable Priority**: Optional constraints to guarantee $\geq 30\%$ children and $\geq 50\%$ women.
3. **Family Cohesion**: Caps the maximum seat allocation per family to ensure broader distribution among independent passenger groups.

---

## 🔌 API Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Single passenger survival inference (with DynamoDB logging) |
| `POST` | `/predict/batch` | Bulk inference for up to 100 passengers |
| `POST` | `/optimize-allocation` | Solve MIP seat allocation for a generated crowd |
| `GET` | `/health` | System health and loaded model check |

---

## 📊 Model Performance

| Algorithm | Accuracy | F1 Score |
| :--- | :--- | :--- |
| **XGBoost (Active)** | **0.85** | **0.82** |
| Random Forest | 0.83 | 0.79 |
| Logistic Regression | 0.81 | 0.76 |

---

## 👤 Author

**Yusuf Musa**
- 📧 [yusuf2000mm@gmail.com](mailto:yusuf2000mm@gmail.com)
- 🔗 [GitHub](https://github.com/yusuuf-mm)

---

## 📄 License
MIT License - Copyright (c) 2026 Yusuf Musa.