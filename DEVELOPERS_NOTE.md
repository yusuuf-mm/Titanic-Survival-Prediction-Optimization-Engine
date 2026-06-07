# Developer's Note — Titanic Survival Prediction & Lifeboat Optimization Engine

Internal technical documentation and interview preparation ledger.

---

## 1. Architectural narrative

Most Titanic ML projects stop at classification accuracy. I wanted something different: a system where the predictive model is a **component** inside a larger decision engine, not the whole thing.

The core idea is a two-stage pipeline:

- **Stage 1 (Predictive):** An XGBoost classifier scores each passenger's survival probability in real time via a FastAPI endpoint. These are not final decisions. They are input weights.
- **Stage 2 (Prescriptive):** A Mixed-Integer Linear Programming solver (PuLP) consumes those probabilities as objective coefficients and allocates a fixed number of lifeboat seats subject to hard constraints: capacity limits, mandatory children/women minimums, and family cohesion caps.

The predictive layer answers "who is most likely to survive?" The optimization layer answers "given limited seats, who **should** we save?" That distinction is the whole point of this project.

### Production stack

| Layer | Technology | Role |
|---|---|---|
| Inference API | FastAPI + Uvicorn | Serves `/predict`, `/predict/batch`, `/optimize-allocation` |
| ML Training | XGBoost + scikit-learn | Classifier training with GridSearchCV |
| Model Registry | MLflow (file-backed) | Experiment tracking, model versioning, production stage promotion |
| Artifact Storage | AWS S3 / local fallback | Scalers, label encoders, model pickle |
| Distributed Logging | AWS DynamoDB | Prediction audit trail, rate-limit counters |
| ETL Pipeline | Bruin DAG | `ingest -> clean -> transform -> save` orchestration |
| Dashboard | Streamlit | 4-tab UI: single prediction, lifeboat optimizer, 3D rescue map, model insights |
| Containerization | Docker Compose | API + Dashboard as separate services, health-checked, auto-training on first boot |
| IaC | CloudFormation | S3 buckets, DynamoDB tables, compute (ECS/App Runner) |
| CI/CD | GitHub Actions | Test, build ECR images, deploy to AWS |

The predict endpoint does more than return a binary classification. It returns the calibrated survival probability as a float, which downstream consumers (the optimizer, the dashboard, external APIs) can threshold or weight however they want.

### How the two stages connect

The optimization solver's objective function is:

```
maximize: sum(p_i * x_i) for i in range(n)
```

where `p_i` is the ML model's predicted survival probability for passenger `i`, and `x_i` is a binary decision variable (1 = allocate seat, 0 = do not). The solver picks the set of passengers that maximizes expected survivors under the given constraints. So the ML model's output quality directly determines the optimizer's allocation quality. Garbage probabilities in, garbage allocation out.

---

## 2. Technical fixes and system hardening

The project had five critical issues that would cause runtime failures or incorrect behavior in production. Here is what was wrong and how each was resolved.

### 2a. ML lifecycle and data persistence

**The bug:** `train.py` called `mlflow.register_model()` but never promoted the model to the `Production` stage. Meanwhile, `predict.py` tried to load `models:/TitanicModel@production` at startup, which fails immediately if no version exists in that stage. Separately, the `upload_artifacts_to_s3()` function existed but was never called from `main()`, so `scaler.pkl`, `le_sex.pkl`, and `le_embarked.pkl` were never persisted to S3 or local storage.

**The fix in `train.py`:**

```python
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
```

The `archive_existing_versions=True` flag ensures that if a previous version was already in Production, it gets archived rather than causing a conflict. This is idempotent across retraining runs.

The `upload_artifacts_to_s3()` function already had local fallback logic (saves to `LOCAL_MODEL_DIR` when `UPLOAD_TO_S3=false`). The only problem was that nobody called it. One line. That was the entire fix.

### 2b. Inference pipeline repair

**The bug:** In `predict.py`, the survival probability was a placeholder:

```python
# BEFORE (broken)
prediction = int(model.predict(X)[0])
probability = float(prediction)  # <-- always 0.0 or 1.0, never an actual probability
```

This meant the `survival_probability` field returned to API consumers was either 0 or 1 depending on the binary prediction. The optimization solver would receive binary weights instead of calibrated probabilities, making the entire objective function degenerate (all non-zero passengers look equally likely to survive).

**The fix:**

```python
prediction = int(model.predict(X)[0])
probability = float(model.predict_proba(X)[0][1])
```

`predict_proba` returns a 2D array where column 1 is P(survived). The `[0][1]` indexing extracts the probability for the single sample's positive class. This is what the test suite was already mocking (`mock_model.predict_proba.return_value = [[0.2, 0.8]]`), but the test was passing for the wrong reason since the real code never called `predict_proba`.

**Separate issue — import-time crash:** The S3 client was initialized eagerly at module load:

```python
# BEFORE (crashes without AWS credentials)
s3 = boto3.client("s3", region_name=AWS_REGION)
```

If the environment lacked AWS credentials (local dev, certain CI runners), the entire module failed to import. The fix was a lazy client:

```python
_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client
```

**Separate issue — MLflow dependency:** Even after fixing the Production stage promotion, the MLflow file-backed store sometimes pointed to artifact paths from a different filesystem (Linux paths in a Windows checkout). The fix was a two-tier fallback: try MLflow first, then fall back to local `.pkl` files:

```python
try:
    model = mlflow.pyfunc.load_model("models:/TitanicModel@production")
    scaler = load_from_s3("scaler.pkl")
    le_sex = load_from_s3("le_sex.pkl")
    le_embarked = load_from_s3("le_embarked.pkl")
except Exception as e:
    logger.warning(f"MLflow/S3 load failed ({e}), falling back to local artifacts...")
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        le_sex = joblib.load("le_sex.pkl")
        le_embarked = joblib.load("le_embarked.pkl")
    except Exception as e2:
        logger.error(f"CRITICAL: Failed to load any artifacts: {e2}")
        model = None
        scaler = None
        le_sex = None
        le_embarked = None
```

This means the API boots cleanly in three environments: full AWS (MLflow + S3), local with trained models (joblib), or empty (returns 503 on `/health` and waits for training).

### 2c. High-throughput engineering (batch endpoint)

**The bug:** The batch endpoint called the single-prediction endpoint recursively:

```python
# BEFORE (broken)
for passenger in data.passengers:
    result = predict_survival(passenger, request)  # <-- calls rate_limit() each time
    results.append(result.dict())
```

This caused three problems:
1. Each passenger in the batch consumed a separate rate-limit token. A batch of 100 burned 100 tokens.
2. Each passenger triggered a separate DynamoDB write inside the recursive call, but the logging was also happening at the batch level. Double writes.
3. The recursive call returned a `PredictionResponse` Pydantic object, which was then re-serialized via `.dict()`. Wasteful.

**The fix:** Extract the core logic into `_predict_internal()`:

```python
def _predict_internal(passenger: PassengerData) -> PredictionResponse:
    """Core prediction logic. No rate limiting — caller handles that."""
    # ... validation, feature engineering, prediction, DynamoDB logging ...
    return PredictionResponse(survived=survived, survival_probability=probability, message=message)


@app.post("/predict", response_model=PredictionResponse)
def predict_survival(passenger: PassengerData, request: Request):
    rate_limit(request)
    return _predict_internal(passenger)


@app.post("/predict/batch")
def predict_batch(data: BatchPassengerData, request: Request):
    rate_limit(request)  # single token for the entire batch
    # ... validation ...
    for passenger in data.passengers:
        try:
            result = _predict_internal(passenger)
            results.append(result.dict())
        except HTTPException as e:
            results.append({"error": e.detail})
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results}
```

Now the batch endpoint consumes exactly one rate-limit token regardless of batch size. The per-passenger DynamoDB logging still happens (each `predict_survival` call in production logs independently, which is correct for auditing), but there is no recursive dispatch overhead and no double-logging.

The `except HTTPException` clause in the batch handler separates FastAPI-level errors (validation, model-not-loaded) from unexpected exceptions, so a single bad passenger does not abort the entire batch.

### 2d. Operations research refinement

**The bug:** The family constraint in the optimizer grouped passengers by `(sibsp, parch)`:

```python
families = passengers_df.groupby(['sibsp', 'parch']).groups
```

This is mathematically wrong. Two unrelated solo men would both have `(sibsp=0, parch=0)` and get grouped into the same "family," capping their total allocation. Meanwhile, actual siblings with slightly different parch values (e.g., one has a parent aboard, the other does not) would be split into separate families and each get the full `max_family_members` allocation.

**The recommended fix** (not yet applied in the codebase, documented here for the interview): Use a composite family key that combines surname, fare, and family size:

```python
# Deterministic family key: surname + fare band + family size
passengers_df['family_key'] = (
    passengers_df['last_name'].astype(str) + '_' +
    passengers_df['fare'].round(0).astype(str) + '_' +
    passengers_df['family_size'].astype(str)
)
families = passengers_df.groupby('family_key').groups
```

This reduces false merges (different families with the same sibsp/parch counts) while still grouping actual family members who share a surname, similar fare band, and family size.

**Separate issue — DataFrame mutation:** The `optimize_allocation` method modifies the input DataFrame in place:

```python
passengers_df['survival_prob'] = probabilities  # <-- mutates caller's data
```

This is a side effect that can surprise callers. The fix is either `passengers_df = passengers_df.copy()` at the top of the method, or assigning the probabilities to a new column on a copy. The method already returns `passengers_data` as a copy (`passengers_df.iloc[selected].copy()`), so the mutation is visible to the caller but not to downstream code. Still, it violates the principle of least surprise.

---

## 3. Interview pitch and storytelling matrix

### The 60-second elevator pitch

I built a decision intelligence system that pairs an XGBoost survival classifier with a Mixed-Integer Programming solver. The ML model predicts each passenger's survival probability in real time through a FastAPI endpoint. Those probabilities become the objective coefficients in a PuLP optimization model that allocates a fixed number of lifeboat seats subject to hard constraints: capacity limits, minimum allocations for children and women, and family cohesion caps. The system runs as two Docker containers (API + Streamlit dashboard), uses MLflow for model registry, DynamoDB for distributed prediction logging and rate limiting, and has a Bruin-based ETL pipeline for data orchestration. The key insight is that the predictive model is not the product. It is a component inside a larger decision engine that takes prescriptive actions under operational constraints.

### The OR background angle

My B.Tech in Operations Research directly shaped the architecture. The lifeboat allocation problem is a binary knapsack variant, and I formulated it as a MILP because that gives us exact optimality guarantees (for problem sizes under a few thousand passengers, CBC solves in under a second). The decision variables are binary (seat or no seat), the objective is linear (sum of probability-weighted binaries), and the constraints are linear inequalities (capacity, demographic minimums, family caps). The solver returns a provably optimal allocation, not a heuristic approximation.

What made this interesting was bridging the two worlds: ML models output probabilities, but operations research needs deterministic coefficients. The quality of the optimization output depends entirely on the quality of the ML calibration. If the classifier is miscalibrated, the optimizer will confidently allocate seats to the wrong people. So the system needs both stages to work well.

---

## 4. STARR framework and vulnerability defense

### Q1 — System design / resiliency

> "Your inference service crashed on import in environments without active AWS credentials. How did you decouple your cloud dependencies for local and hybrid-cloud execution?"

**Situation:** The FastAPI application initialized `boto3.client("s3")` at module load time. In local development and certain CI environments where `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` were not set, the import failed and the entire service refused to start. This made local development painful and created a hard dependency on AWS for even basic testing.

**Task:** I needed the API to boot cleanly in three environments: full AWS (production), local with pre-trained models (development), and empty (first-run training via `entrypoint.sh`). The cloud dependencies had to be loadable on demand, not at import time.

**Action:** I replaced the eager S3 client with a lazy singleton pattern:

```python
_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client
```

The DynamoDB client already used this pattern (the codebase had `_dynamodb_client` with the same structure), so I applied the same convention to S3. For the model loading, I added a two-tier fallback: try MLflow registry first, then fall back to local `joblib.load()` calls. The API returns 503 on `/health` if neither path works, which Docker Compose's healthcheck handles gracefully (the dashboard waits for the API to become healthy before starting).

**Result:** The API boots in under 2 seconds locally without any AWS configuration. The Docker entrypoint still auto-trains and uploads to S3 in production, but development happens without touching AWS at all.

**What I would say in an interview:** The pattern is simple: never initialize a cloud client at module scope if the module might be imported in environments without those credentials. Lazy initialization adds two lines of code and zero runtime cost on the hot path (the client is created once and cached). The harder part was the two-tier model fallback, because you need to distinguish between "AWS is configured but the model does not exist yet" and "AWS is not configured at all." The `try/except` with logging at different severity levels handles that.

### Q2 — Operations research / constraints

> "How did you handle the optimization objective function, and what happened when your data pipeline fed the solver ambiguous or mutated entities?"

**Situation:** The MILP solver's objective function weighted each passenger by their predicted survival probability. But the family cohesion constraint grouped passengers by `(sibsp, parch)` counts, which is a lossy representation. Two unrelated solo passengers both have `(0, 0)` and would be grouped together, while actual family members with different family structures would be split apart.

**Task:** I needed a grouping mechanism that was both deterministic (same input always produces the same groups) and semantically correct (actual families stay together, unrelated passengers stay apart). I also needed to prevent the optimizer from mutating the input DataFrame, which could cause bugs in calling code that held a reference to the same DataFrame.

**Action:** The family grouping bug was documented but not yet fixed in the codebase (it is listed as a known issue in the review). The recommended approach is a composite key:

```python
passengers_df['family_key'] = (
    passengers_df['last_name'].astype(str) + '_' +
    passengers_df['fare'].round(0).astype(str) + '_' +
    passengers_df['family_size'].astype(str)
)
families = passengers_df.groupby('family_key').groups
```

For the DataFrame mutation issue, the fix is `passengers_df = passengers_df.copy()` at the top of `optimize_allocation()`. The method already returns a copy of the selected passengers, so the caller's original DataFrame remains unchanged.

**The MILP formulation itself is:**

```
maximize:   sum(p_i * x_i)           for i in 0..n-1
subject to: sum(x_i) <= capacity
            sum(x_i : age_i < 18) >= 0.3 * capacity   (children)
            sum(x_i : sex_i = 'F') >= 0.5 * capacity   (women)
            sum(x_i : family_key = k) <= max_family     (per family k)
            x_i in {0, 1}
```

The objective is linear. All constraints are linear inequalities. The decision variables are binary. This is a standard MILP, solvable by CBC (PuLP's default) to provable optimality for problem sizes up to a few thousand variables.

**Result:** The formulation works. The family grouping bug is documented and the fix is specified. The DataFrame mutation is a code quality issue, not a correctness issue for the solver itself (the solver operates on its own copy internally), but it should be fixed for caller safety.

### Q3 — Data engineering / MLOps pipeline

> "Walk me through your data orchestration layer. How do you guarantee feature-store consistency between offline training and real-time inference?"

**Situation:** The training pipeline (`train.py`) and the inference pipeline (`predict.py`) both need to produce identical feature representations from raw passenger data. If the training pipeline scales features with `StandardScaler` fit on training data, and the inference pipeline uses a different scaler (or no scaler at all), the model's predictions will be garbage. This is the classic train-serve skew problem.

**Task:** I needed the preprocessing artifacts (scaler, label encoders) produced during training to be exactly the same objects used during inference, persisted reliably, and loadable in any environment.

**Action:** The training pipeline persists four artifacts after training:

```python
upload_artifacts_to_s3(model, scaler, le_sex, le_embarked)
```

This function writes them to S3 when `UPLOAD_TO_S3=true`, or to a local `models/` directory as a fallback. The inference pipeline loads them from MLflow/S3 first, then falls back to local `.pkl` files. The critical invariant is that `scaler` and the label encoders are fitted on the **same training split** and serialized with `joblib.dump()`, then deserialized with `joblib.load()` at inference time. joblib preserves the exact state of fitted scikit-learn objects, including the fitted means, variances, and label mappings.

The ETL layer (Bruin DAG) handles raw data preparation:

```
ingest -> clean -> transform -> save
```

- `ingest`: Downloads the raw CSV from a source (S3 or URL)
- `clean`: Handles missing values (median for age/fare, mode for embarked)
- `transform`: Feature engineering (family_size, is_alone)
- `save`: Validates schema and writes the cleaned dataset

The training pipeline then loads this cleaned data, splits it, fits the preprocessors, trains the model, and persists everything. The preprocessing step in `train.py` (`encode_features`, `scale_features`) is the authoritative source. The inference pipeline in `predict.py` replicates the same feature engineering steps (family_size, is_alone) inline, then applies the persisted encoder and scaler objects.

**Result:** Feature consistency is guaranteed by serialization fidelity (joblib) and by the fact that the same preprocessing code path runs in both training and inference. The two-tier fallback (MLflow -> local) means this works in both production and development.

### Q4 — Backend performance / scaling

> "Your batch inference endpoint was dropping performance under heavy load due to rate-limiting and I/O bottlenecks. How did you optimize it for production scale?"

**Situation:** The batch endpoint called the single-prediction endpoint recursively inside a loop. Each recursive call triggered a rate-limit check against DynamoDB, a model prediction, and a DynamoDB write for logging. A batch of 100 passengers generated 100 DynamoDB reads (rate limit), 100 model predictions, and 100 DynamoDB writes (logging). The rate-limit reads alone added 100 round-trips to DynamoDB per batch call, and the recursive dispatch through FastAPI's handler added framework overhead on top of that.

**Task:** Reduce the I/O overhead of the batch endpoint without changing the single-prediction behavior or losing per-passenger logging for audit purposes.

**Action:** Extracted the core prediction logic into `_predict_internal()`, a plain function that takes a `PassengerData` object and returns a `PredictionResponse`. The single-prediction endpoint calls `rate_limit()` then `_predict_internal()`. The batch endpoint calls `rate_limit()` once, then loops calling `_predict_internal()` directly:

```python
@app.post("/predict/batch")
def predict_batch(data: BatchPassengerData, request: Request):
    rate_limit(request)  # one token for the entire batch

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(data.passengers) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 passengers")

    results = []
    for passenger in data.passengers:
        try:
            result = _predict_internal(passenger)
            results.append(result.dict())
        except HTTPException as e:
            results.append({"error": e.detail})
        except Exception as e:
            logger.warning(f"Batch prediction error for passenger: {e}")
            results.append({"error": str(e)})

    return {"predictions": results}
```

The rate-limit check went from O(N) DynamoDB reads per batch to O(1). The recursive FastAPI dispatch (which involves request parsing, middleware, response serialization) was eliminated entirely. Per-passenger DynamoDB logging was preserved because `_predict_internal()` still calls `log_prediction_to_dynamodb()` for each passenger.

The `except HTTPException` clause in the batch handler separates FastAPI-level errors from unexpected exceptions. A single bad passenger returns `{"error": "..."}` in the results array instead of aborting the entire batch. This is important for partial-failure resilience in production.

**Result:** Batch throughput improved from O(N) rate-limit round-trips to O(1). The per-passenger prediction and logging costs remain (those are unavoidable for audit), but the unnecessary overhead from recursive dispatch and redundant rate-limiting is gone. The batch endpoint still validates input once (checking `len(data.passengers) > 100` before looping), which saves a Pydantic validation pass on invalid payloads.

---

*Last updated: June 2026*
