# End-to-End Customer Churn

Predicts customer churn on the Telco dataset: data validation, feature engineering, a calibrated XGBoost model with a cost-based decision threshold, MLflow tracking, a FastAPI service with a small web UI, and drift monitoring.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```powershell
python Scripts/run_pipeline.py --input data/OG/Telco-Customer-Churn.csv --tune --n_trials 15
```

Useful options: `--cost_fp` / `--cost_fn` (cost of a wasted retention offer / a missed churner, used to pick the threshold), `--threshold` (fix it instead), `--cv_folds`. Without `--tune` the default hyperparameters are used.

Outputs:
- `artifacts/` — `model.joblib`, `explain_model.pkl`, `feature_schema.json`, `threshold.json`, `model_metadata.json` (everything the API loads)
- `mlflow.db` + `mlruns/` — experiment tracking (`mlflow ui --backend-store-uri sqlite:///mlflow.db`)

## Serve

```powershell
uvicorn src.api.main:app --port 8000
```

- UI: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs
- `GET /health` shows which training run the API is serving
- `POST /predict` returns the churn probability, the decision, and the top drivers

Quick check: `python Scripts/test_fastapi.py`

### Docker

```powershell
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

Train first so `artifacts/` exists; the image copies it in.

## Test

```powershell
pytest
```

Includes a parity test that checks the serving encoder produces exactly the same features as training for every row. API tests are skipped until the model has been trained.

## Monitor

Every prediction is appended to `logs/predictions.jsonl`. Compare recent traffic against the training data:

```powershell
python Scripts/check_drift.py
```

It prints a PSI score per feature (stable < 0.1, significant > 0.25) and exits with code 1 if drift is significant.

## Layout

```
Scripts/    run_pipeline.py (train), check_drift.py, prepare_processed_data.py, test_fastapi.py
src/data/       load + preprocess
src/features/   build_features.py (training encoding and the shared serving encoder)
src/models/     train.py, tune.py, evaluate.py
src/utils/      validate_data.py (Great Expectations checks)
src/api/        FastAPI app + static UI
tests/          pytest suite
```
