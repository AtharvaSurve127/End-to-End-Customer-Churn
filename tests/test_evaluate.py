import numpy as np

from src.models.evaluate import choose_threshold, probability_metrics


def test_threshold_drops_when_misses_are_expensive():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(0.5 * y + rng.normal(0.25, 0.2, 2000), 0, 1)
    cheap_miss, _ = choose_threshold(y, proba, cost_fp=1, cost_fn=1)
    costly_miss, _ = choose_threshold(y, proba, cost_fp=1, cost_fn=10)
    assert costly_miss < cheap_miss


def test_threshold_is_cost_minimum():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.9])
    best, curve = choose_threshold(y, proba, cost_fp=1, cost_fn=1)
    assert curve[best] == 0
    assert curve[best] == min(curve.values())


def test_probability_metrics_perfect_predictor():
    m = probability_metrics(np.array([0, 1, 0, 1]), np.array([0.0, 1.0, 0.0, 1.0]))
    assert m["roc_auc"] == 1.0
    assert m["brier"] == 0.0
