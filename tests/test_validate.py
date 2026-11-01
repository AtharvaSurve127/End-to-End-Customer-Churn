from src.utils.validate_data import validate_telco_data


def test_real_dataset_passes(raw_df):
    is_valid, failed = validate_telco_data(raw_df)
    assert is_valid, failed


def test_bad_churn_value_fails(raw_df):
    df = raw_df.copy()
    df.loc[0, "Churn"] = "Maybe"
    is_valid, failed = validate_telco_data(df)
    assert not is_valid
    assert any("Churn" in f for f in failed)


def test_negative_tenure_fails(raw_df):
    df = raw_df.copy()
    df.loc[0, "tenure"] = -1
    is_valid, failed = validate_telco_data(df)
    assert not is_valid
    assert any("tenure" in f for f in failed)


def test_missing_column_fails(raw_df):
    is_valid, failed = validate_telco_data(raw_df.drop(columns=["Contract"]))
    assert not is_valid
    assert any("Contract" in f for f in failed)
