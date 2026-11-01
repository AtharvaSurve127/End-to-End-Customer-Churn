import json
import os
import sys
from datetime import datetime, timezone
from typing import Literal

import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.features.build_features import encode_for_serving

ARTIFACTS_DIR = os.getenv("CHURN_ARTIFACTS_DIR", os.path.join(PROJECT_ROOT, "artifacts"))
PREDICTION_LOG = os.getenv("PREDICTION_LOG", os.path.join(PROJECT_ROOT, "logs", "predictions.jsonl"))

app = FastAPI(title="Telco Churn API")

YesNo = Literal["Yes", "No"]
YesNoNoPhone = Literal["Yes", "No", "No phone service"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: YesNoNoPhone
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


def _read_json(name: str) -> dict:
    with open(os.path.join(ARTIFACTS_DIR, name)) as f:
        return json.load(f)


# Artifacts written by Scripts/run_pipeline.py
SCHEMA = _read_json("feature_schema.json")
METADATA = _read_json("model_metadata.json")
MODEL = joblib.load(os.path.join(ARTIFACTS_DIR, "model.joblib"))
EXPLAIN_MODEL = joblib.load(os.path.join(ARTIFACTS_DIR, "explain_model.pkl"))
# Decision threshold chosen by expected cost during training
THRESHOLD = float(os.getenv("CHURN_THRESHOLD") or _read_json("threshold.json")["threshold"])


def _top_drivers(X: pd.DataFrame, k: int = 5):
    """Per-feature contributions (log-odds) from the XGBoost booster; positive pushes toward churn."""
    contribs = EXPLAIN_MODEL.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    ranked = sorted(zip(X.columns, contribs, X.iloc[0]), key=lambda t: abs(t[1]), reverse=True)[:k]
    return [
        {"feature": name, "value": float(val), "contribution": round(float(c), 3),
         "effect": "raises churn risk" if c > 0 else "lowers churn risk"}
        for name, c, val in ranked
    ]


def _log_prediction(inputs: dict, proba: float, prediction: int):
    """Append one prediction to a JSONL file for monitoring (Scripts/check_drift.py)."""
    try:
        os.makedirs(os.path.dirname(PREDICTION_LOG), exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "inputs": inputs,
            "churn_probability": proba,
            "churn_prediction": prediction,
            "run_id": METADATA["run_id"],
        }
        with open(PREDICTION_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # monitoring must never break a prediction


@app.get("/health")
def health():
    return {
        "status": "ok",
        "n_features": len(SCHEMA["feature_columns"]),
        "run_id": METADATA["run_id"],
        "trained_at": METADATA["trained_at"],
        "threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(customer: Customer):
    inputs = customer.model_dump()
    try:
        X = encode_for_serving(pd.DataFrame([inputs]), SCHEMA)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    proba = float(MODEL.predict_proba(X)[:, 1][0])
    prediction = int(proba >= THRESHOLD)
    _log_prediction(inputs, proba, prediction)
    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": prediction,
        "threshold": THRESHOLD,
        "top_drivers": _top_drivers(X),
    }


# Serve the web UI at "/" (mounted last so API routes take precedence)
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="ui")
