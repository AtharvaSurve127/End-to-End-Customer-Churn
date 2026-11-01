from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

DEFAULT_PARAMS = dict(
    n_estimators=301, learning_rate=0.034, max_depth=7,
    subsample=0.95, colsample_bytree=0.98,
    n_jobs=-1, random_state=42, eval_metric="logloss",
)


def make_model(params: dict) -> CalibratedClassifierCV:
    """
    XGBoost wrapped in isotonic calibration so predicted probabilities are real
    percentages. Class imbalance is handled by the cost-based decision threshold
    rather than scale_pos_weight, which would inflate the probabilities.
    """
    return CalibratedClassifierCV(XGBClassifier(**params), method="isotonic", cv=5)


def out_of_fold_proba(params: dict, X, y, folds: int = 5):
    """Out-of-fold churn probabilities on the training data (used for CV metrics and the threshold)."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return cross_val_predict(make_model(params), X, y, cv=skf, method="predict_proba")[:, 1]


def train_final(params: dict, X, y):
    """
    Fit the calibrated model used for predictions, plus a plain XGBoost model
    used only to explain predictions (per-feature contributions).
    """
    model = make_model(params).fit(X, y)
    explain_model = XGBClassifier(**params).fit(X, y)
    return model, explain_model
