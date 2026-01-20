#!/usr/bin/env python3
"""
Titanic Survival Prediction API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Optional
import os

# Initialize FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict passenger survival on the Titanic using machine learning",
    version="1.0.0"
)

# Load models and preprocessors
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_sex = joblib.load('le_sex.pkl')
    le_embarked = joblib.load('le_embarked.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None

# Input schema
class PassengerData(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    sex: str = Field(..., description="Sex (male or female)")
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Passenger fare")
    embarked: str = Field(..., description="Port of embarkation (C, Q, or S)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pclass": 3,
                "sex": "male",
                "age": 22.0,
                "sibsp": 1,
                "parch": 0,
                "fare": 7.25,
                "embarked": "S"
            }
        }

# Output schema
class PredictionResponse(BaseModel):
    survived: int
    survival_probability: float
    message: str

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
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_survival(passenger: PassengerData):
    """
    Predict whether a passenger would survive the Titanic disaster
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
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
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        # Prepare response
        survived = int(prediction)
        message = "Likely to survive" if survived == 1 else "Unlikely to survive"
        
        return PredictionResponse(
            survived=survived,
            survival_probability=float(probability),
            message=message
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
class BatchPassengerData(BaseModel):
    passengers: list[PassengerData]

@app.post("/predict/batch")
def predict_batch(data: BatchPassengerData):
    """
    Make predictions for multiple passengers
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for passenger in data.passengers:
        try:
            result = predict_survival(passenger)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)