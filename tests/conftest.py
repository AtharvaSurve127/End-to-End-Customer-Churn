import os

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_CSV = os.path.join(ROOT, "data", "OG", "Telco-Customer-Churn.csv")


@pytest.fixture(scope="session")
def raw_df():
    if not os.path.exists(RAW_CSV):
        pytest.skip("raw dataset not available")
    return pd.read_csv(RAW_CSV)


@pytest.fixture
def sample_customer():
    return {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 350.75,
    }
