#!/usr/bin/env python3
"""
Titanic Survival Prediction - Training Script
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

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

def save_artifacts(model, scaler, le_sex, le_embarked, output_dir='.'):
    """Save model and preprocessing objects"""
    print("\nSaving model and preprocessing objects...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(output_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(le_sex, os.path.join(output_dir, 'le_sex.pkl'))
    joblib.dump(le_embarked, os.path.join(output_dir, 'le_embarked.pkl'))
    
    print("Saved:")
    print("  - model.pkl")
    print("  - scaler.pkl")
    print("  - le_sex.pkl")
    print("  - le_embarked.pkl")

def main():
    """Main training pipeline"""
    print("="*60)
    print("TITANIC SURVIVAL PREDICTION - TRAINING PIPELINE")
    print("="*60)
    
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
    accuracy = evaluate_model(model, X_test_scaled, y_test)
    
    # Save
    save_artifacts(model, scaler, le_sex, le_embarked)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()