#!/bin/bash

# Deploy CloudFormation stacks in order: storage, compute, api

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
        STACK_PREFIX=$(yq eval '.stack_prefix' ../../config/aws/config.yaml)
    else
        echo "Warning: yq not found, using grep/awk (less robust)"
        AWS_REGION=$(grep 'region:' ../../config/aws/config.yaml | awk '{print $2}')
        STACK_PREFIX=$(grep 'stack_prefix:' ../../config/aws/config.yaml | awk '{print $2}')
    fi
else
    echo "Error: AWS config not found"
    exit 1
fi

# Deploy storage stack
aws cloudformation deploy \
    --template-file ../../infra/cloudformation/storage-stack.yaml \
    --stack-name ${STACK_PREFIX}-storage \
    --region $AWS_REGION \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides Environment=$ENV

# Deploy compute stack
aws cloudformation deploy \
    --template-file ../../infra/cloudformation/compute-stack.yaml \
    --stack-name ${STACK_PREFIX}-compute \
    --region $AWS_REGION \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides Environment=$ENV

# Deploy api stack
aws cloudformation deploy \
    --template-file ../../infra/cloudformation/api-stack.yaml \
    --stack-name ${STACK_PREFIX}-api \
    --region $AWS_REGION \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides Environment=$ENV

echo "All stacks deployed successfully"