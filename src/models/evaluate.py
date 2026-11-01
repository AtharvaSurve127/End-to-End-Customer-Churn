import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def probability_metrics(y_true, proba) -> dict:
    """Threshold-independent metrics. Brier score measures calibration (lower is better)."""
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "avg_precision": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def baseline_scores(X_train, y_train, X_test, y_test) -> dict:
    """Score trivial and linear baselines so the main model has something to beat."""
    baselines = {
        "majority_class": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
    }
    scores = {}
    for name, model in baselines.items():
        model.fit(X_train, y_train)
        scores[name] = probability_metrics(y_test, model.predict_proba(X_test)[:, 1])
    return scores


def choose_threshold(y_true, proba, cost_fp: float, cost_fn: float):
    """
    Pick the threshold with the lowest expected cost.

    cost_fp: cost of contacting a customer who would have stayed (retention offer).
    cost_fn: cost of missing a customer who churns (lost revenue).
    Returns (best_threshold, {threshold: cost}).
    """
    y_true = np.asarray(y_true)
    curve = {}
    for t in np.round(np.arange(0.02, 0.99, 0.01), 2):
        pred = proba >= t
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        curve[float(t)] = fp * cost_fp + fn * cost_fn
    best = min(curve, key=curve.get)
    return best, curve


def calibration_figure(y_true, proba, title="Calibration (test set)"):
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
