import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.preprocess import preprocess_data
from src.features.build_features import (
    build_features, encode_for_serving, get_column_groups,
)


@pytest.fixture(scope="module")
def trained(raw_df):
    df = preprocess_data(raw_df.copy())
    binary_cols, multi_cols = get_column_groups(df, "Churn")
    enc = build_features(df, target_col="Churn")
    X = enc.drop(columns=["Churn"])
    schema = {"feature_columns": list(X.columns), "binary_cols": binary_cols, "multi_cols": multi_cols}
    return df, X, schema


def test_column_groups(trained):
    df, _, schema = trained
    assert "gender" in schema["binary_cols"]
    assert "Contract" in schema["multi_cols"]
    assert "Churn" not in schema["binary_cols"] + schema["multi_cols"]


def test_features_are_numeric(trained):
    _, X, _ = trained
    assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
    assert X.isna().sum().sum() == 0


def test_serving_matches_training_for_every_row(trained):
    """The core guarantee: serving encodes rows exactly as training did."""
    df, X, schema = trained
    served = encode_for_serving(df.drop(columns=["Churn"]), schema)
    assert list(served.columns) == list(X.columns)
    assert_frame_equal(served, X.astype(float), check_dtype=False)


def test_serving_single_row_matches_batch(trained):
    df, _, schema = trained
    batch = encode_for_serving(df.drop(columns=["Churn"]), schema)
    single = encode_for_serving(df.drop(columns=["Churn"]).iloc[[10]], schema)
    assert_frame_equal(single, batch.iloc[[10]])


def test_serving_unseen_category_is_ignored(trained):
    df, _, schema = trained
    row = df.drop(columns=["Churn"]).iloc[[0]].copy()
    row["Contract"] = "Five year"
    out = encode_for_serving(row, schema)
    assert out[[c for c in out.columns if c.startswith("Contract_")]].sum().sum() == 0


def test_serving_rejects_invalid_binary_value(trained):
    df, _, schema = trained
    row = df.drop(columns=["Churn"]).iloc[[0]].copy()
    row["gender"] = "male"
    with pytest.raises(ValueError, match="gender"):
        encode_for_serving(row, schema)
