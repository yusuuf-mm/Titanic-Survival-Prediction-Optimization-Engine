# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY predict.py .
COPY train.py .
COPY entrypoint.sh .
COPY data/ ./data/
COPY optimization/ ./optimization/

# Copy pre-trained model artifacts (if present; entrypoint will auto-train if missing)
COPY model.pk[l] scaler.pk[l] le_sex.pk[l] le_embarked.pk[l] ./

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://localhost:8000/health')"

# Run entrypoint (auto-trains if models missing, then starts API)
ENTRYPOINT ["bash", "/app/entrypoint.sh"]