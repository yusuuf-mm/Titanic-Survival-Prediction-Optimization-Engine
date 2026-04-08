#!/bin/bash

# Setup script: configure AWS CLI, create S3 bucket, upload artifacts

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
        S3_BUCKET=$(yq eval '.s3_bucket' ../../config/aws/config.yaml)
    else
        echo "Warning: yq not found, using grep/awk (less robust)"
        AWS_REGION=$(grep 'region:' ../../config/aws/config.yaml | awk '{print $2}')
        S3_BUCKET=$(grep 's3_bucket:' ../../config/aws/config.yaml | awk '{print $2}')
    fi
else
    echo "Error: AWS config not found"
    exit 1
fi

# Configure AWS CLI
aws configure set region $AWS_REGION
aws configure set output json

# Create S3 bucket if it doesn't exist
if ! aws s3api head-bucket --bucket $S3_BUCKET 2>/dev/null; then
    aws s3 mb s3://$S3_BUCKET --region $AWS_REGION
    echo "S3 bucket $S3_BUCKET created"
else
    echo "S3 bucket $S3_BUCKET already exists"
fi

# Upload model artifacts
aws s3 cp ../../model.pkl s3://$S3_BUCKET/models/model.pkl
aws s3 cp ../../scaler.pkl s3://$S3_BUCKET/models/scaler.pkl
aws s3 cp ../../le_sex.pkl s3://$S3_BUCKET/models/le_sex.pkl
aws s3 cp ../../le_embarked.pkl s3://$S3_BUCKET/models/le_embarked.pkl

echo "Model artifacts uploaded to S3"