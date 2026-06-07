#!/usr/bin/env python3
"""
Titanic Survival Prediction - Training Script
"""

import mlflow

mlflow.set_tracking_uri("file:./mlruns")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import boto3
import logging
import mlflow.xgboost

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables for configuration
S3_BUCKET = os.getenv('S3_BUCKET_NAME', 'titanic-prediction-bucket')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Lazy S3 client initialization
_s3_client = None

def get_s3_client():
    """Lazy initialization of S3 client"""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3', region_name=AWS_REGION)
    return _s3_client

def load_and_preprocess_data(filepath='data/titanic.csv'):
    """Load and preprocess the Titanic dataset"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Feature engineering
    print("Engineering features...")
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    
    # Handle missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    df['fare'].fillna(df['fare'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    
    # Select features
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size', 'is_alone']
    X = df[features].copy()
    y = df['survived']
    
    return X, y

def encode_features(X_train, X_test):
    """Encode categorical features"""
    print("Encoding categorical features...")
    
    # Label encoders
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    # Fit and transform training data
    X_train['sex'] = le_sex.fit_transform(X_train['sex'])
    X_train['embarked'] = le_embarked.fit_transform(X_train['embarked'])
    
    # Transform test data
    X_test['sex'] = le_sex.transform(X_test['sex'])
    X_test['embarked'] = le_embarked.transform(X_test['embarked'])
    
    return X_train, X_test, le_sex, le_embarked

def scale_features(X_train, X_test):
    """Scale numerical features"""
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train, tune_hyperparameters=True):
    """Train XGBoost model with optional hyperparameter tuning"""
    
    if tune_hyperparameters:
        print("Training with hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    else:
        print("Training with default parameters...")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy

def upload_artifacts_to_s3(model, scaler, le_sex, le_embarked):
    """Upload model and preprocessing objects to S3, with local fallback"""
    # Check if S3 upload should be attempted
    upload_to_s3 = os.getenv('UPLOAD_TO_S3', 'true').lower() == 'true'
    
    if not upload_to_s3:
        logger.info("S3 upload disabled (UPLOAD_TO_S3=false), skipping...")
        # Save locally as fallback
        _save_artifacts_locally(model, scaler, le_sex, le_embarked)
        return
    
    logger.info("Uploading model and preprocessing objects to S3...")

    artifacts = {
        'model.pkl': model,
        'scaler.pkl': scaler,
        'le_sex.pkl': le_sex,
        'le_embarked.pkl': le_embarked
    }

    for filename, artifact in artifacts.items():
        try:
            # Save to temporary file first
            joblib.dump(artifact, filename)
            # Upload to S3 using lazy client
            s3 = get_s3_client()
            s3.upload_file(filename, S3_BUCKET, filename)
            # Clean up temporary file
            os.remove(filename)
            logger.info(f"Uploaded {filename} to S3")
        except Exception as e:
            logger.error(f"Failed to upload {filename} to S3: {e}")
            logger.info("Falling back to local storage...")
            # Save locally instead
            _save_artifacts_locally(model, scaler, le_sex, le_embarked)
            return

    logger.info("All artifacts uploaded successfully to S3")

def _save_artifacts_locally(model, scaler, le_sex, le_embarked):
    """Save artifacts to local directory as fallback"""
    artifacts = {
        'model.pkl': model,
        'scaler.pkl': scaler,
        'le_sex.pkl': le_sex,
        'le_embarked.pkl': le_embarked
    }
    
    local_dir = os.getenv('LOCAL_MODEL_DIR', 'models')
    os.makedirs(local_dir, exist_ok=True)
    
    for filename, artifact in artifacts.items():
        filepath = os.path.join(local_dir, filename)
        joblib.dump(artifact, filepath)
        logger.info(f"Saved {filename} locally to {filepath}")

def main():
    """Main training pipeline"""
    logger.info("Starting Titanic Survival Prediction - Training Pipeline")
    
    with mlflow.start_run():
        max_depth = 5
        n_estimators = 200
        learning_rate = 0.1
        
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        
        # Load data
        X, y = load_and_preprocess_data()
        
        # Split data
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Encode features
        X_train, X_test, le_sex, le_embarked = encode_features(X_train, X_test)
        
        # Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Train model
        model = train_model(X_train_scaled, y_train, tune_hyperparameters=False)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        evaluate_model(model, X_test_scaled, y_test)
        
        # Log model
        mlflow.xgboost.log_model(model, "model")

        # Register model and promote to Production
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        registered = mlflow.register_model(model_uri, "TitanicModel")
        mlflow.tracking.MlflowClient().transition_model_version_stage(
            name="TitanicModel",
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"Model registered and promoted to Production (version {registered.version})")

        # Persist preprocessing artifacts (needed by predict.py / optimization)
        upload_artifacts_to_s3(model, scaler, le_sex, le_embarked)

        logger.info(f"Training complete! Final Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()