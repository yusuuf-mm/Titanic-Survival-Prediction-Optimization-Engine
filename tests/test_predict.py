import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
import joblib
import pandas as pd
import numpy as np

# Mocking S3 before importing the app to avoid real AWS calls during initialization
with patch('boto3.client') as mock_boto:
    # Set up the mock S3 client
    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3
    
    # Pre-emptively import predict.app after patching
    from predict import app, load_from_s3

client = TestClient(app)

@patch('predict.get_s3_client')
def test_load_from_s3(mock_get_s3_client):
    mock_s3_client = mock_get_s3_client.return_value
    # Mock S3 response
    mock_body = MagicMock()
    mock_body.read.return_value = b'fake_model_data'
    mock_s3_client.get_object.return_value = {'Body': mock_body}
    
    # Mock joblib.load
    with patch('joblib.load') as mock_joblib_load:
        mock_joblib_load.return_value = "mocked_artifact"
        
        artifact = load_from_s3("fake_key.pkl")
        
        assert artifact == "mocked_artifact"
        mock_s3_client.get_object.assert_called_once()

def test_health_check_unloaded():
    """Test health check when model is not loaded"""
    with patch('predict.model', None):
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not loaded"

@patch('predict.model')
@patch('predict.scaler')
@patch('predict.le_sex')
@patch('predict.le_embarked')
def test_predict_success(mock_le_embarked, mock_le_sex, mock_scaler, mock_model):
    """Test successful prediction with mocked models"""
    # Configure mocks
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]
    mock_le_sex.transform.return_value = [1]
    mock_le_embarked.transform.return_value = [2]
    mock_scaler.transform.return_value = np.zeros((1, 9))

    payload = {
        "pclass": 3,
        "sex": "male",
        "age": 22.0,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S"
    }
    
    # Mock DynamoDB logging to avoid AWS calls
    with patch('predict.log_prediction_to_dynamodb'):
        response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["survived"] == 1
    assert data["survival_probability"] == 0.8
    assert "Likely to survive" in data["message"]

def test_predict_invalid_input():
    """Test prediction with invalid input"""
    payload = {
        "pclass": 4, # Invalid class
        "sex": "male",
        "age": -5,   # Invalid age
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "X" # Invalid port
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Pydantic validation error or our ValueError
