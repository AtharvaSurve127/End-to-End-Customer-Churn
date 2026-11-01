import json
import os

import pytest

from conftest import ROOT

ARTIFACTS = os.path.join(ROOT, "artifacts")
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ARTIFACTS, "model.joblib")),
    reason="train the model first: python Scripts/run_pipeline.py --input data/OG/Telco-Customer-Churn.csv",
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["PREDICTION_LOG"] = str(tmp_path_factory.mktemp("logs") / "predictions.jsonl")
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_features"] > 0
    assert 0 < body["threshold"] < 1


def test_predict_response_shape(client, sample_customer):
    r = client.post("/predict", json=sample_customer)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_prediction"] == int(body["churn_probability"] >= body["threshold"])
    assert 1 <= len(body["top_drivers"]) <= 5


def test_risky_customer_scores_higher_than_loyal_one(client, sample_customer):
    loyal = {**sample_customer, "tenure": 70, "Contract": "Two year",
             "InternetService": "DSL", "PaymentMethod": "Credit card (automatic)",
             "MonthlyCharges": 55.0, "TotalCharges": 3850.0}
    risky = client.post("/predict", json=sample_customer).json()["churn_probability"]
    safe = client.post("/predict", json=loyal).json()["churn_probability"]
    assert risky > safe


@pytest.mark.parametrize("field,value", [
    ("gender", "string"), ("gender", "male"), ("Contract", "Weekly"),
    ("tenure", -1), ("MonthlyCharges", -5), ("SeniorCitizen", 2),
])
def test_invalid_values_are_rejected(client, sample_customer, field, value):
    assert client.post("/predict", json={**sample_customer, field: value}).status_code == 422


def test_missing_field_is_rejected(client, sample_customer):
    del sample_customer["tenure"]
    assert client.post("/predict", json=sample_customer).status_code == 422


def test_prediction_is_logged(client, sample_customer):
    client.post("/predict", json=sample_customer)
    with open(os.environ["PREDICTION_LOG"]) as f:
        record = json.loads(f.readlines()[-1])
    assert record["inputs"]["tenure"] == 5
    assert "churn_probability" in record


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Churn" in r.text
