# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commonly Used Commands

### Development
- `python train.py` - Train XGBoost model with feature engineering (family size, embarked encoding)
- `python optimize.py` - Run lifeboat allocation optimization using PuLP solver
- `uvicorn predict:app --reload` - Launch FastAPI service with auto-reload
- `python predict.py [JSON_INPUT]` - Make single prediction via CLI
- `cd optimization && python lifeboat_optimization.py` - Run OR component directly

### Testing (after test files are added)
- `pytest tests/` - Run tests once unit/integration tests are implemented
- `coverage run -m pytest` - Generate coverage report

### Linting
- `flake8 .` - Check code style (assuming flake8 is in requirements.txt)
- `black .` - Auto-format Python code (if black is installed)

### Deployment
- `docker build -t titanic-api .` - Build Docker image
- `docker run -d -p 8000:8000 titanic-api` - Run containerized API
- `gcloud builds submit` - Deploy to Google Cloud (requires setup)

## Code Architecture

### Main Components
1. **Data Pipeline**
   - `data/titanic.csv`: Central dataset with passenger features
   - `le_sex.pkl`/`le_embarked.pkl`: Label encoders for categorical features
   - `scaler.pkl`: StandardScaler for numerical features

2. **Model Training**
   - `train.py`: Handles data preprocessing, model training (XGBoost), and metric tracking
   - `model.pkl`: Trained model artifact with best hyperparameters

3. **API Service**
   - `predict.py`: FastAPI endpoint for individual predictions
   - OpenAPI docs at `/docs` for interactive API exploration

4. **Optimization Module**
   - `optimization/lifeboat_optimization.py`: Uses survival probabilities from model to solve linear program
   - Prescribes ethical resource allocation under capacity constraints

### Key Files for New Development
- `requirements.txt`: Contains Python dependencies including xgboost, fastapi, pulp
- `Dockerfile`: Defines container image for deployment
- `README.md`: Full project documentation

## Critical Context

1. The XGBoost model achieves **85% accuracy** with survival probability calibration
2. OR optimization demonstrates **ethical resource allocation** (prioritizing women/children)
3. All model artifacts (scaler, encoders, model) are persisted separately for reproducibility
4. API expects specific input format with required fields: pclass, sex, age, sibsp, parch, fare, embarked

## Recommended First Actions
- Add test files in a `tests/` directory to enable automated validation
- Implement input validation in the API to handle malformed requests
- Consider adding unit tests for the OR solver components
- Verify Docker build process locally before cloud deployment