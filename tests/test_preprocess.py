import pandas as pd

from src.data.preprocess import preprocess_data


def _raw():
    return pd.DataFrame({
        " customerID ": ["a", "b", "c"],
        "SeniorCitizen": [0, 1, None],
        "TotalCharges": ["10.5", " ", "30"],
        "Churn": ["Yes", "No", " No "],
    })


def test_drops_id_and_strips_headers():
    out = preprocess_data(_raw())
    assert "customerID" not in out.columns


def test_target_mapped_to_binary():
    out = preprocess_data(_raw())
    assert out["Churn"].tolist() == [1, 0, 0]


def test_total_charges_blank_becomes_zero():
    out = preprocess_data(_raw())
    assert out["TotalCharges"].tolist() == [10.5, 0.0, 30.0]


def test_senior_citizen_is_int_without_nan():
    out = preprocess_data(_raw())
    assert out["SeniorCitizen"].tolist() == [0, 1, 0]
    assert str(out["SeniorCitizen"].dtype).startswith("int")


def test_real_dataset_has_no_nans_after_preprocess(raw_df):
    out = preprocess_data(raw_df.copy())
    assert out.isna().sum().sum() == 0
    assert set(out["Churn"].unique()) == {0, 1}
