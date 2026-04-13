#!/bin/bash
set -e

echo "============================================"
echo " Titanic Prediction API — Startup"
echo "============================================"

# Auto-train models if .pkl files are missing
if [ ! -f /app/model.pkl ] || [ ! -f /app/scaler.pkl ] || [ ! -f /app/le_sex.pkl ] || [ ! -f /app/le_embarked.pkl ]; then
    echo ">> Model artifacts not found. Running training pipeline..."
    python /app/train.py
    
    # Move artifacts from models/ to /app/ if training saved them there
    if [ -d /app/models ] && [ -f /app/models/model.pkl ]; then
        echo ">> Moving trained artifacts to /app/..."
        cp /app/models/*.pkl /app/
    fi
    
    echo ">> Training complete."
else
    echo ">> Model artifacts found. Skipping training."
fi

echo ">> Starting FastAPI server..."
exec uvicorn predict:app --host 0.0.0.0 --port 8000
