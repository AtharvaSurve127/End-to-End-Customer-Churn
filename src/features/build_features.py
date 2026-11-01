import pandas as pd

# Deterministic binary encodings. Training (build_features) and serving
# (encode_for_serving) both read these, so they cannot drift apart.
YES_NO_MAP = {"No": 0, "Yes": 1}
GENDER_MAP = {"Female": 0, "Male": 1}


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.

    Yes/No and Male/Female use the fixed maps above. Any other 2-category
    feature falls back to alphabetical ordering.
    """
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    if valset == set(YES_NO_MAP):
        return s.map(YES_NO_MAP).astype("Int64")

    if valset == set(GENDER_MAP):
        return s.map(GENDER_MAP).astype("Int64")

    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    return s


def get_column_groups(df: pd.DataFrame, target_col: str = "Churn"):
    """Split categorical columns into (binary_cols, multi_cols) by cardinality."""
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    return binary_cols, multi_cols


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Feature engineering for training data: binary encoding for 2-category
    columns, one-hot encoding (drop_first=True) for the rest.

    Serving must reproduce this exactly; see encode_for_serving.
    """
    df = df.copy()
    binary_cols, multi_cols = get_column_groups(df, target_col)
    print(f"Feature engineering: {len(binary_cols)} binary, {len(multi_cols)} multi-category columns")

    for c in binary_cols:
        df[c] = _map_binary_series(df[c].astype(str))

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Nullable Int64 -> plain int for XGBoost
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)

    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)

    print(f"Feature engineering complete: {df.shape[1]} columns")
    return df


def encode_for_serving(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Encode raw (preprocessed, target-free) rows into the training feature matrix.

    schema comes from training: {"feature_columns", "binary_cols", "multi_cols"}.
    One-hot columns are created without drop_first and then reindexed to the
    training columns, which drops the reference level and fills unseen
    categories with 0. Works for a single row or a batch.
    """
    df = df.copy()

    for c in schema["binary_cols"]:
        mapping = GENDER_MAP if c == "gender" else YES_NO_MAP
        encoded = df[c].map(mapping)
        if encoded.isna().any():
            bad = df.loc[encoded.isna(), c].unique().tolist()
            raise ValueError(f"Invalid value(s) for {c}: {bad}")
        df[c] = encoded

    df = pd.get_dummies(df, columns=schema["multi_cols"])
    return df.reindex(columns=schema["feature_columns"], fill_value=0).astype(float)
