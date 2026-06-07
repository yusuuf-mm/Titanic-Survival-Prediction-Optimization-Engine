#!/usr/bin/env python3
"""
Titanic Survival Prediction API
"""

import mlflow
import mlflow.xgboost
mlflow.set_tracking_uri("file:./mlruns")

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Optional
import boto3
import io
import os
import logging
from datetime import datetime
from functools import wraps
import time
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS Configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

# Lazy S3 client initialization
_s3_client = None

def get_s3_client():
    """Lazy initialization of S3 client"""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client

def load_from_s3(key):
    """Clean helper function to load artifacts from S3"""
    obj = get_s3_client().get_object(Bucket=S3_BUCKET_NAME, Key=key)
    return joblib.load(io.BytesIO(obj["Body"].read()))

def get_dynamodb_client():
    """Lazy initialization of DynamoDB client"""
    global _dynamodb_client
    if _dynamodb_client is None:
        _dynamodb_client = boto3.client('dynamodb', region_name=AWS_REGION)
    return _dynamodb_client

# Initialize FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict passenger survival on the Titanic using machine learning",
    version="1.0.0"
)


# Environment variables for features
ENABLE_RATE_LIMIT = os.getenv('ENABLE_RATE_LIMIT', 'false').lower() == 'true'

def getDynamodbTable():
    return os.getenv('DYNAMODB_TABLE_NAME', 'predictions')

def getRateLimitTable():
    return os.getenv('RATE_LIMIT_TABLE', 'rate-limits')

# Lazy DynamoDB initialization
_dynamodb_client = None

# MLflow Model Registry
MODEL_NAME = "TitanicModel"

# Load model from MLflow Model Registry, other artifacts from S3
# Falls back to local .pkl files if MLflow/S3 are unavailable
try:
    logger.info(f"Loading model from MLflow Model Registry: {MODEL_NAME}")
    model = mlflow.pyfunc.load_model("models:/TitanicModel@production")
    logger.info("Model loaded from MLflow Model Registry successfully.")

    logger.info(f"Loading remaining artifacts from S3 bucket: {S3_BUCKET_NAME}")
    scaler = load_from_s3("scaler.pkl")
    le_sex = load_from_s3("le_sex.pkl")
    le_embarked = load_from_s3("le_embarked.pkl")
    logger.info("Artifacts successfully loaded.")
except Exception as e:
    logger.warning(f"MLflow/S3 load failed ({e}), falling back to local artifacts...")
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        le_sex = joblib.load("le_sex.pkl")
        le_embarked = joblib.load("le_embarked.pkl")
        logger.info("Local artifact fallback succeeded.")
    except Exception as e2:
        logger.error(f"CRITICAL: Failed to load any artifacts: {e2}")
        model = None
        scaler = None
        le_sex = None
        le_embarked = None

# Input schema
class PassengerData(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    sex: str = Field(..., description="Sex (male or female)")
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Passenger fare")
    embarked: str = Field(..., description="Port of embarkation (C, Q, or S)")

    model_config = {"json_schema_extra": {
            "example": {
                "pclass": 3,
                "sex": "male",
                "age": 22.0,
                "sibsp": 1,
                "parch": 0,
                "fare": 7.25,
                "embarked": "S"
            }
        }}

# Output schema
class PredictionResponse(BaseModel):
    survived: int
    survival_probability: float
    message: str

def log_prediction_to_dynamodb(passenger_data, prediction, probability):
    """Log prediction to DynamoDB"""
    try:
        client = get_dynamodb_client()
        item = {
            'id': {'S': str(int(time.time() * 1000000))},  # Unique ID
            'timestamp': {'S': datetime.utcnow().isoformat()},
            'pclass': {'N': str(passenger_data.pclass)},
            'sex': {'S': passenger_data.sex},
            'age': {'N': str(passenger_data.age)},
            'sibsp': {'N': str(passenger_data.sibsp)},
            'parch': {'N': str(passenger_data.parch)},
            'fare': {'N': str(passenger_data.fare)},
            'embarked': {'S': passenger_data.embarked},
            'survived': {'N': str(prediction)},
            'survival_probability': {'N': str(probability)}
        }
        client.put_item(TableName=getDynamodbTable(), Item=item)
        logger.info("Prediction logged to DynamoDB")
    except Exception as e:
        logger.error(f"Failed to log prediction to DynamoDB: {e}")

def rate_limit(request: Request):
    """Rate limiting using DynamoDB for distributed/multi-worker environments
    
    Uses fail-closed approach for security - if rate limiting fails,
    requests are denied to prevent abuse. Includes automatic TTL cleanup.
    Can be disabled via ENABLE_RATE_LIMIT env var.
    """
    # Skip rate limiting if disabled
    if not ENABLE_RATE_LIMIT:
        return
    
    client_ip = request.headers.get('X-Forwarded-For', request.client.host)
    current_window = int(time.time() / 60)  # 1-minute windows
    rate_key = f"ratelimit:{client_ip}:{current_window}"
    current_time = int(time.time())
    
    # Configurable rate limit - default 60 requests per minute
    rate_limit_max = int(os.getenv('RATE_LIMIT_MAX', '60'))
    
    try:
        dynamodb = get_dynamodb_client()
        
        # Try to get current count from DynamoDB
        response = dynamodb.get_item(
            TableName=getRateLimitTable(),
            Key={'id': {'S': rate_key}}
        )
        
        current_count = 0
        if 'Item' in response:
            # Check if entry has expired
            expires_at = int(response['Item'].get('expires', {}).get('N', 0))
            if expires_at > 0 and current_time > expires_at:
                # Entry expired, delete it and start fresh
                dynamodb.delete_item(
                    TableName=getRateLimitTable(),
                    Key={'id': {'S': rate_key}}
                )
                current_count = 0
            else:
                current_count = int(response['Item']['count']['N'])
        
        # Check if rate limit exceeded
        if current_count >= rate_limit_max:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
        
        # Increment counter
        dynamodb.put_item(
            TableName=getRateLimitTable(),
            Item={
                'id': {'S': rate_key},
                'count': {'N': str(current_count + 1)},
                'expires': {'N': str(current_time + 60)}
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        # Fail-closed: deny request if rate limiting fails for security
        logger.error(f"Rate limiting failed (fail-closed): {e}")
        raise HTTPException(status_code=503, detail="Rate limiting temporarily unavailable")

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make a survival prediction",
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive API documentation"
        }
    }

# Health check
@app.get("/health")
def health_check():
    if model is None:
        logger.error("Health check failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    logger.info("Health check passed")
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
def _predict_internal(passenger: PassengerData) -> PredictionResponse:
    """Core prediction logic. No rate limiting — caller handles that."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Additional input validation
        if passenger.fare < 0:
            raise ValueError("Fare cannot be negative")
        if passenger.age < 0 or passenger.age > 120:
            raise ValueError("Age must be between 0 and 120")
        if passenger.sex.lower() not in ['male', 'female']:
            raise ValueError("Sex must be 'male' or 'female'")
        if passenger.embarked.upper() not in ['C', 'Q', 'S']:
            raise ValueError("Embarked must be 'C', 'Q', or 'S'")

        # Feature engineering
        family_size = passenger.sibsp + passenger.parch + 1
        is_alone = 1 if family_size == 1 else 0

        # Create feature dictionary
        features = {
            'pclass': passenger.pclass,
            'sex': passenger.sex,
            'age': passenger.age,
            'sibsp': passenger.sibsp,
            'parch': passenger.parch,
            'fare': passenger.fare,
            'embarked': passenger.embarked,
            'family_size': family_size,
            'is_alone': is_alone
        }

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Encode categorical variables
        df['sex'] = le_sex.transform(df['sex'])
        df['embarked'] = le_embarked.transform(df['embarked'])

        # Scale features
        X = scaler.transform(df)

        # Make prediction
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        # Prepare response
        survived = prediction
        message = "Likely to survive" if survived == 1 else "Not likely to survive"

        # Log prediction to DynamoDB
        log_prediction_to_dynamodb(passenger, survived, probability)

        return PredictionResponse(
            survived=survived,
            survival_probability=probability,
            message=message
        )

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict_survival(passenger: PassengerData, request: Request):
    """
    Predict whether a passenger would survive the Titanic disaster
    """
    rate_limit(request)
    return _predict_internal(passenger)


# Batch prediction endpoint
class BatchPassengerData(BaseModel):
    passengers: list[PassengerData]

@app.post("/predict/batch")
def predict_batch(data: BatchPassengerData, request: Request):
    """
    Make predictions for multiple passengers (single rate-limit token).
    """
    rate_limit(request)

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(data.passengers) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 passengers")

    results = []
    for passenger in data.passengers:
        try:
            result = _predict_internal(passenger)
            results.append(result.dict())
        except HTTPException as e:
            results.append({"error": e.detail})
        except Exception as e:
            logger.warning(f"Batch prediction error for passenger: {e}")
            results.append({"error": str(e)})

    logger.info(f"Batch prediction completed for {len(data.passengers)} passengers")
    return {"predictions": results}

from optimization.lifeboat_optimization import LifeboatOptimizer
try:
    optimizer = LifeboatOptimizer()
except Exception as e:
    logger.error(f"Failed to initialize LifeboatOptimizer: {e}")
    optimizer = None

class OptimizationRequest(BaseModel):
    passengers: list[dict]
    capacity: int
    priority_children: bool = True
    priority_women: bool = True
    max_family_members: Optional[int] = None

@app.post("/optimize-allocation")
def optimize_allocation_endpoint(req: OptimizationRequest, request: Request):
    """
    Optimize lifeboat allocation
    """
    rate_limit(request)
    
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not loaded")
        
    try:
        df = pd.DataFrame(req.passengers)
        results = optimizer.optimize_allocation(
            df, 
            req.capacity, 
            priority_children=req.priority_children, 
            priority_women=req.priority_women, 
            max_family_members=req.max_family_members
        )
        
        # Convert passengers_data dataframe back to dicts for JSON
        res_dict = {
            'status': results['status'],
            'objective_value': results['objective_value'],
            'selected_passengers': results['passengers_data'].to_dict(orient='records'),
            'selected_count': results['selected_count'],
            'capacity': results['capacity'],
            'utilization': results['utilization']
        }
        return res_dict
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)