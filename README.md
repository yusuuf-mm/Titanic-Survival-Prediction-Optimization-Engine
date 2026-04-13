# Titanic Survival Prediction & Lifeboat Optimization Engine

A premium, full-stack decision intelligence system that fuses **Machine Learning** predictions with **Operations Research** optimization. This project not only predicts passenger survival but also solves the critical resource allocation problem: "How to maximize survivors under strict lifeboat capacity and ethical constraints?"

![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Docker](https://img.shields.io/badge/Docker-enabled-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🌟 Key Features

- 🔮 **ML Prediction API**: High-performance survival classifier powered by **XGBoost**.
- 🧮 **Optimization Engine**: Linear Programming (LP) solver using **PuLP** to allocate limited lifeboat seats based on predictive risk and ethical priority rules.
- 🎨 **Executive Dashboard**: A stunning, modern **Streamlit** UI for real-time inference and optimization visualization.
- 🐳 **One-Command Deployment**: Fully containerized with **Docker Compose**, including auto-training capabilities.
- 🚀 **Auto-Recovery**: Intelligent entrypoint that automatically trains models on startup if artifacts are missing.

---

## 🏗️ Project Architecture

```
Titanic-Optimization-Engine/
│
├── api/ (Root)
│   ├── predict.py             # FastAPI prediction & optimization endpoints
│   ├── train.py               # XGBoost training pipeline
│   ├── entrypoint.sh          # Intelligent startup & auto-training script
│   └── model.pkl              # Trained artifacts (auto-generated)
│
├── dashboard.py               # Streamlit Executive Dashboard
│
├── optimization/
│   ├── __init__.py            # Module initialization
│   └── lifeboat_optimization.py # LP Optimization logic
│
├── data/
│   └── titanic.csv            # Historical passenger data
│
├── Dockerfile                 # API Container definition
├── Dockerfile.dashboard       # Dashboard Container definition
└── docker-compose.yml         # Full system orchestration
```

---

## 🚀 Quick Start (Recommended)

The entire system is integrated. You don't need to manually train models or install local dependencies if you have Docker.

### 1. Start the Engine
```bash
docker compose up --build -d
```

### 2. Access the System
- **Executive Dashboard**: `http://localhost:8501`
- **Inference API**: `http://localhost:8000/docs`

---

## 💻 Manual Setup (Development)

### 1. Installation
```bash
git clone https://github.com/yusuuf-mm/Titanic-Survival-Prediction-Optimization-Engine.git
cd Titanic-Survival-Prediction-Optimization-Engine
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Training
```bash
python train.py
```

### 3. Run Services
```bash
# Terminal 1: API
uvicorn predict:app --reload

# Terminal 2: Dashboard
streamlit run dashboard.py
```

---

## 🧮 The Optimization Problem

We treat lifeboat allocation as a **Mixed-Integer Programming (MIP)** problem.

**Objective:**
Maximize $\sum (p_i \cdot x_i)$ where $p_i$ is the predicted survival probability of passenger $i$, and $x_i$ is the binary decision to allocate a seat.

**Constraints Support:**
1.  **Capacity**: $\sum x_i \leq \text{Available Seats}$
2.  **Vulnerable Priority**: Guarantees $\geq 30\%$ children and $\geq 50\%$ women.
3.  **Family Cohesion**: Limits seat allocation per family to ensure broader distribution.

---

## 🔌 API Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Single passenger survival inference |
| `POST` | `/predict/batch` | Bulk inference for up to 100 passengers |
| `POST` | `/optimize-allocation` | Solve LP seat allocation for a crowd |
| `GET` | `/health` | System health check |

---

## 📊 Model Performance

| Algorithm | Accuracy | F1 Score | AUC |
| :--- | :--- | :--- | :--- |
| **XGBoost (Active)** | **0.85** | **0.82** | **0.91** |
| Random Forest | 0.83 | 0.79 | 0.89 |
| Logistic Regression | 0.81 | 0.76 | 0.86 |

---

## 👤 Author

**Yusuf Musa**
- 📧 [yusuf2000mm@gmail.com](mailto:yusuf2000mm@gmail.com)
- 🔗 [GitHub](https://github.com/yusuuf-mm)

---

## 📄 License
MIT License - Copyright (c) 2026 Yusuf Musa.