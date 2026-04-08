# Cloud-Native Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Titanic Survival Prediction Optimization Engine to AWS using Infrastructure as Code (IaC) with CloudFormation. The deployment creates a serverless architecture leveraging AWS Lambda, API Gateway, ECS Fargate, S3, and DynamoDB.

## Prerequisites

### AWS Account Setup

- Active AWS account with billing enabled
- AWS CLI installed and configured (`aws --version` to verify)
- Python 3.10+ installed (`python --version` to verify)
- Docker installed and running (`docker --version` to verify)

### IAM Permissions Required

The following IAM permissions are required for deployment:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cloudformation:*",
        "s3:*",
        "ecr:*",
        "lambda:*",
        "apigateway:*",
        "ecs:*",
        "iam:*",
        "dynamodb:*",
        "logs:*"
      ],
      "Resource": "*"
    }
  ]
}
```

### Environment Configuration

- Set the `ENV` environment variable to either `dev` or `prod`
- Ensure AWS credentials are configured via `aws configure`

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                         AWS Cloud Architecture                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐ │
│  │   Client     │───▶│   API Gateway    │───▶│    Lambda/ECS      │ │
│  │  (Requests)  │    │  (/predict,      │    │  (FastAPI)        │ │
│  │              │    │   /optimize)     │    │                   │ │
│  └──────────────┘    └──────────────────┘    └─────────┬─────────┘ │
│                                                        │           │
│                                                        ▼           │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐ │
│  │  DynamoDB    │◀───│   Prediction     │◀───│       S3          │ │
│  │  (Logs,     │    │   Logging         │    │  (Models)         │ │
│  │   Rate Limit)│    │                   │    │  - model.pkl      │ │
│  │              │    │                   │    │  - scaler.pkl     │ │
│  └──────────────┘    └──────────────────┘    │  - le_*.pkl        │ │
│                                              └───────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Prediction Request** → API Gateway → Lambda/ECS → S3 (load model) → DynamoDB (log result)
2. **Model Training** → Local training → S3 (upload artifacts)
3. **Rate Limiting** → DynamoDB (distributed counter with TTL)

### Service Justifications

- **API Gateway**: Provides RESTful API interface with authentication, throttling, and monitoring
- **Lambda**: Serverless compute for single predictions - cost-effective, auto-scaling, no server management
- **ECS Fargate**: Containerized batch processing for optimization tasks requiring more compute resources
- **S3**: Object storage for ML models and artifacts - durable, versioned, cost-effective
- **DynamoDB**: NoSQL database for prediction logging - fast, scalable, pay-per-use

## Project Structure

```
titanic-survival-prediction-optimization-engine/
├── config/
│   ├── aws/
│   │   └── config.yaml          # AWS configuration
│   └── environment/
│       ├── dev.env              # Development environment vars
│       └── prod.env             # Production environment vars
├── deploy/
│   ├── Dockerfile               # Container definition
│   └── scripts/
│       ├── setup.sh             # AWS setup and model upload
│       ├── build.sh             # Docker build and ECR push
│       └── deploy.sh            # CloudFormation deployment
├── infra/
│   └── cloudformation/
│       ├── storage-stack.yaml   # S3 + DynamoDB
│       ├── compute-stack.yaml  # Lambda + ECS
│       └── api-stack.yaml      # API Gateway
├── docs/
│   ├── deployment-guide.md     # This guide
│   └── api-docs.md             # API documentation
├── optimization/
│   └── lifeboat_optimization.py # OR optimization logic
├── data/
│   └── titanic.csv             # Training dataset
├── notebooks/
│   ├── eda.ipynb              # Exploratory analysis
│   ├── training.ipynb         # Model training
│   └── optimization.ipynb     # OR analysis
├── predict.py                 # FastAPI prediction service
├── train.py                   # Model training script
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview
```

## Deployment Steps

### Step 0: Train and Upload Model (Local)

First, train the model locally and upload artifacts to S3:

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py
```

**Note:** `train.py` automatically uploads the trained model artifacts to S3.

### Step 1: Initial Setup

Run the setup script to configure AWS resources:

```bash
# Set environment (dev or prod)
export ENV=dev

# Run setup script
cd deploy/scripts
chmod +x setup.sh
./setup.sh
```

**What this script does:**

- Loads environment-specific configuration from `config/environment/${ENV}.env`
- Configures AWS CLI region
- Creates S3 bucket (if not exists)
- Creates DynamoDB tables (predictions, rate limits)

### Step 2: Build Container Image

Build and push the Docker image to Amazon ECR:

```bash
# Run build script
cd deploy/scripts
chmod +x build.sh
./build.sh
```

**What this script does:**

- Authenticates Docker with ECR
- Builds the application container
- Tags with timestamp
- Pushes to ECR repository

### Step 3: Deploy Infrastructure

Deploy the CloudFormation stacks in order:

```bash
# Run deployment script
cd deploy/scripts
chmod +x deploy.sh
./deploy.sh
```

**What this script does:**

- Deploys storage stack (S3 + DynamoDB)
- Deploys compute stack (Lambda + ECS)
- Deploys API stack (API Gateway)

## Testing the Deployment

### Get API Gateway URL

After deployment, retrieve the API Gateway URL:

```bash
aws cloudformation describe-stacks \
    --stack-name titanic-prediction-api \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
    --output text
```

### Test Health Check

```bash
curl -X GET "https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/health"
```

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Test Single Prediction

```bash
curl -X POST "https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "pclass": 3,
       "sex": "male",
       "age": 25.0,
       "sibsp": 0,
       "parch": 0,
       "fare": 7.25,
       "embarked": "S"
     }'
```

Expected response:

```json
{
  "survived": 0,
  "survival_probability": 0.123,
  "message": "Unlikely to survive"
}
```

### Test Batch Prediction

```bash
curl -X POST "https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "passengers": [
         {
           "pclass": 1,
           "sex": "female",
           "age": 29.0,
           "sibsp": 1,
           "parch": 0,
           "fare": 100.0,
           "embarked": "C"
         },
         {
           "pclass": 3,
           "sex": "male",
           "age": 22.0,
           "sibsp": 0,
           "parch": 0,
           "fare": 7.25,
           "embarked": "S"
         }
       ]
     }'
```

## Cost Estimates (AWS Free Tier)

The deployment is designed to stay within AWS Free Tier limits for development:

| Service     | Free Tier Limit                   | Estimated Monthly Cost |
| ----------- | --------------------------------- | ---------------------- |
| Lambda      | 1M requests + 400,000 GB-seconds  | $0 (within limits)     |
| API Gateway | 1M requests                       | $0 (within limits)     |
| S3          | 5GB storage + 20,000 GET requests | $0 (within limits)     |
| DynamoDB    | 25GB storage + 200M requests      | $0 (within limits)     |
| ECS Fargate | 2,500 CPU hours + 1,250 GB-hours  | $0 (within limits)     |
| ECR         | 500MB storage                     | $0 (within limits)     |

**Total Estimated Cost:** $0/month (Free Tier eligible)

_Note: Costs may vary based on usage. Monitor via AWS Cost Explorer._

## Scalability Considerations

### Horizontal Scaling

- **Lambda**: Automatically scales to handle concurrent requests
- **ECS**: Can scale tasks based on demand using CloudWatch alarms
- **DynamoDB**: Auto-scaling enabled for read/write capacity

### Performance Optimization

- **Lambda**: Provisioned concurrency for cold start reduction
- **API Gateway**: Response caching and throttling
- **S3**: CloudFront CDN for model artifact distribution

### Monitoring

- CloudWatch metrics and logs for all services
- X-Ray for distributed tracing
- API Gateway access logging

## Troubleshooting

### Common Issues

1. **CloudFormation Stack Creation Fails**
   - Check IAM permissions
   - Verify AWS region configuration
   - Check CloudFormation events for specific errors

2. **Lambda Function Errors**
   - Check CloudWatch logs for the function
   - Verify environment variables
   - Ensure model artifacts exist in S3

3. **API Gateway 403 Errors**
   - Check API Gateway execution role permissions
   - Verify Lambda function policies

### Logs and Monitoring

```bash
# View CloudFormation stack events
aws cloudformation describe-stack-events --stack-name titanic-prediction-api

# View Lambda function logs
aws logs tail /aws/lambda/TitanicPredictionFunction --follow

# View ECS task logs
aws logs tail /ecs/titanic-optimization-task --follow
```

## Cleanup

To remove all deployed resources:

```bash
# Delete stacks in reverse order
aws cloudformation delete-stack --stack-name titanic-prediction-api
aws cloudformation delete-stack --stack-name titanic-prediction-compute
aws cloudformation delete-stack --stack-name titanic-prediction-storage

# Remove ECR images
aws ecr batch-delete-image --repository-name titanic-prediction --image-ids imageTag=latest

# Empty and delete S3 bucket
aws s3 rm s3://titanic-prediction-bucket --recursive
aws s3 rb s3://titanic-prediction-bucket
```

---

This deployment provides a production-ready, scalable, and cost-effective solution for the Titanic Survival Prediction service with Operations Research optimization capabilities.
