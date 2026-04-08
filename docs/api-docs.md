# API Documentation

## Overview

The Titanic Survival Prediction API provides RESTful endpoints for predicting passenger survival and performing batch optimization. Built with FastAPI, the API serves machine learning predictions and operations research optimization results.

**Base URL:** `https://your-api-gateway-id.execute-api.region.amazonaws.com/prod`

**Authentication:** AWS IAM (for production) or none (development)

## Endpoints

### GET /

**Description:** API information and available endpoints

**Response:**

```json
{
  "message": "Titanic Survival Prediction API",
  "version": "1.0.0",
  "endpoints": {
    "/predict": "POST - Make a survival prediction",
    "/health": "GET - Check API health",
    "/docs": "GET - Interactive API documentation"
  }
}
```

### GET /health

**Description:** Health check endpoint to verify service status and model loading

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Error Responses:**

- `503 Service Unavailable`: Model not loaded

### POST /predict

**Description:** Predict survival probability for a single passenger

**Request Headers:**

```
Content-Type: application/json
```

**Request Body:**

```json
{
  "pclass": 3,
  "sex": "male",
  "age": 25.0,
  "sibsp": 0,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S"
}
```

**Field Descriptions:**

| Field      | Type    | Required | Description                | Constraints        |
| ---------- | ------- | -------- | -------------------------- | ------------------ |
| `pclass`   | integer | Yes      | Passenger class            | 1-3                |
| `sex`      | string  | Yes      | Gender                     | "male" or "female" |
| `age`      | float   | Yes      | Age in years               | 0-120              |
| `sibsp`    | integer | Yes      | Number of siblings/spouses | ≥ 0                |
| `parch`    | integer | Yes      | Number of parents/children | ≥ 0                |
| `fare`     | float   | Yes      | Passenger fare             | ≥ 0                |
| `embarked` | string  | Yes      | Port of embarkation        | "C", "Q", or "S"   |

**Response:**

```json
{
  "survived": 0,
  "survival_probability": 0.123,
  "message": "Unlikely to survive"
}
```

**Response Fields:**

| Field                  | Type    | Description                                    |
| ---------------------- | ------- | ---------------------------------------------- |
| `survived`             | integer | Binary prediction (0 = perished, 1 = survived) |
| `survival_probability` | float   | Probability of survival (0.0 to 1.0)           |
| `message`              | string  | Human-readable prediction result               |

### POST /predict/batch

**Description:** Predict survival for multiple passengers (max 100)

**Request Headers:**

```
Content-Type: application/json
```

**Request Body:**

```json
{
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
}
```

**Response:**

```json
{
  "predictions": [
    {
      "survived": 1,
      "survival_probability": 0.987,
      "message": "Likely to survive"
    },
    {
      "survived": 0,
      "survival_probability": 0.089,
      "message": "Unlikely to survive"
    }
  ]
}
```

## Error Codes

### HTTP Status Codes

| Code  | Description           | Common Causes                            |
| ----- | --------------------- | ---------------------------------------- |
| `200` | Success               | Request processed successfully           |
| `400` | Bad Request           | Invalid input data, validation errors    |
| `429` | Too Many Requests     | Rate limit exceeded (10 requests/minute) |
| `500` | Internal Server Error | Prediction processing error              |
| `503` | Service Unavailable   | Model not loaded or service unavailable  |

### Error Response Format

All error responses follow this structure:

```json
{
  "detail": "Error message description"
}
```

### Specific Error Examples

**Validation Error (400):**

```json
{
  "detail": "Invalid input: Age must be between 0 and 120"
}
```

**Rate Limit Exceeded (429):**

```json
{
  "detail": "Rate limit exceeded. Try again later."
}
```

**Model Not Loaded (503):**

```json
{
  "detail": "Model not loaded"
}
```

**Prediction Error (500):**

```json
{
  "detail": "Prediction error: [specific error details]"
}
```

## Rate Limiting

- **Limit:** 10 requests per minute per IP address
- **Scope:** Applies to all endpoints
- **Implementation:** Simple counter (production should use Redis)

## Data Validation

### Input Validation Rules

1. **Passenger Class (pclass):**
   - Must be integer 1, 2, or 3

2. **Sex:**
   - Must be exactly "male" or "female" (case-sensitive)

3. **Age:**
   - Must be float between 0 and 120 (inclusive)

4. **Siblings/Spouses (sibsp) and Parents/Children (parch):**
   - Must be non-negative integers

5. **Fare:**
   - Must be non-negative float

6. **Embarked:**
   - Must be exactly "C", "Q", or "S" (case-sensitive)

7. **Batch Size:**
   - Maximum 100 passengers per batch request

## Feature Engineering

The API automatically performs feature engineering:

- **Family Size:** `sibsp + parch + 1`
- **Is Alone:** `1` if family_size == 1, else `0`

## Logging

All predictions are logged to DynamoDB with the following information:

- Unique ID (timestamp-based)
- All input features
- Prediction result
- Survival probability
- Timestamp

## Testing Examples

### Using curl

**Single Prediction:**

```bash
curl -X POST "https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "pclass": 1,
       "sex": "female",
       "age": 29.0,
       "sibsp": 1,
       "parch": 0,
       "fare": 100.0,
       "embarked": "C"
     }'
```

**Batch Prediction:**

```bash
curl -X POST "https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "passengers": [
         {"pclass": 1, "sex": "female", "age": 35.0, "sibsp": 1, "parch": 0, "fare": 53.1, "embarked": "S"},
         {"pclass": 3, "sex": "male", "age": 25.0, "sibsp": 0, "parch": 0, "fare": 7.05, "embarked": "S"}
       ]
     }'
```

### Using Python requests

```python
import requests

# Single prediction
url = "https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/predict"
data = {
    "pclass": 2,
    "sex": "female",
    "age": 28.0,
    "sibsp": 1,
    "parch": 0,
    "fare": 26.0,
    "embarked": "S"
}

response = requests.post(url, json=data)
print(response.json())
```

## Interactive Documentation

Visit `/docs` endpoint for interactive Swagger UI documentation with built-in request testing.

---

**Version:** 1.0.0
**Last Updated:** 2024
**Framework:** FastAPI
**Runtime:** Python 3.10+</content>
</xai:function_call">{"path":"docs/api-docs.md","operation":"created","notice":"You do not need to re-read the file, as you have seen all changes Proceed with the task using these changes as the new baseline."}
<environment_details>

# VSCode Visible Files

docs\api-docs.md

# VSCode Open Tabs

requirements.txt,predict.py,train.py,optimization/lifeboat_optimization.py,infra/cloudformation/storage-stack.yaml,infra/cloudformation/compute-stack.yaml,infra/cloudformation/api-stack.yaml,deploy/Dockerfile,deploy/scripts/build.sh,deploy/scripts/deploy.sh,deploy/scripts/setup.sh,docs/deployment-guide.md,docs/api-docs.md,config/environment/dev.env,config/environment/prod.env,config/aws/config.yaml

# Current Time

Current time in ISO 8601 UTC format: 2026-04-07T22:56:24.048Z
User time zone: Africa/Lagos, UTC+1:00

# Current Cost

$0.00

# Current Mode

<slug>code</slug>
<name>Code</name>
<model>x-ai/grok-code-fast-1:optimized:free</model>
<tool_format>native</tool_format>
You have not created a todo list yet. Create one with `update_todo_list` if your task is complicated or involves multiple steps.
</environment_details>  
<xai:function_call name="update_todo_list">
<parameter name="todos">["Gather project information and understand architecture","[x] Create docs/deployment-guide.md with all required sections","[x] Create docs/api-docs.md with API details","Verify files created and contents are correct"]
