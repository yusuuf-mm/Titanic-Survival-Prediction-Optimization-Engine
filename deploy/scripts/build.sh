#!/bin/bash

# Build Docker image, tag and push to ECR

set -e  # Exit on error

# Load environment variables
if [ -f "../../config/environment/${ENV}.env" ]; then
    source "../../config/environment/${ENV}.env"
else
    echo "Error: Environment file not found"
    exit 1
fi

# Load AWS config
if [ -f "../../config/aws/config.yaml" ]; then
    # Try yq first, fall back to grep if not available
    if command -v yq &> /dev/null; then
        AWS_REGION=$(yq eval '.region' ../../config/aws/config.yaml)
        ECR_REPO=$(yq eval '.ecr_repo' ../../config/aws/config.yaml)
    else
        echo "Warning: yq not found, using grep/awk (less robust)"
        AWS_REGION=$(grep 'region:' ../../config/aws/config.yaml | awk '{print $2}')
        ECR_REPO=$(grep 'ecr_repo:' ../../config/aws/config.yaml | awk '{print $2}')
    fi
else
    echo "Error: AWS config not found"
    exit 1
fi

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO

# Build Docker image
docker build -f ../Dockerfile -t titanic-prediction .

# Tag the image
IMAGE_TAG=$(date +%Y%m%d%H%M%S)
docker tag titanic-prediction:latest $ECR_REPO/titanic-prediction:$IMAGE_TAG

# Push to ECR
docker push $ECR_REPO/titanic-prediction:$IMAGE_TAG

echo "Image pushed: $ECR_REPO/titanic-prediction:$IMAGE_TAG"