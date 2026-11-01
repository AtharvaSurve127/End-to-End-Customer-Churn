#!/usr/bin/env python3
"""
Compare recent API traffic (logs/predictions.jsonl) against the training data.

Reports the Population Stability Index (PSI) for every input feature, plus the
average predicted churn probability against the training churn rate.
PSI < 0.1: stable | 0.1-0.25: moderate shift | > 0.25: significant shift.

Usage:
    python Scripts/check_drift.py [--log logs/predictions.jsonl] [--min_rows 30]
Exit code is 1 if any feature shows significant drift.
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
EPS = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI between two arrays of bin shares (each sums to 1)."""
    e, a = np.clip(expected, EPS, None), np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref.dropna(), edges)[0] / ref.notna().sum()
    a = np.histogram(cur.dropna(), edges)[0] / cur.notna().sum()
    return psi(e, a)


def categorical_psi(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna()) | set(cur.dropna()))
    e = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    a = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return psi(e, a)


def label(value: float) -> str:
    return "stable" if value < 0.1 else "moderate" if value < 0.25 else "SIGNIFICANT"


def main(args):
    if not os.path.exists(args.log):
        sys.exit(f"No prediction log at {args.log}. Serve some predictions first.")
    with open(args.log) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if len(records) < args.min_rows:
        sys.exit(f"Only {len(records)} logged predictions; need at least {args.min_rows} for a meaningful comparison.")

    cur = pd.DataFrame([r["inputs"] for r in records])
    ref = pd.read_csv(args.reference)
    ref["TotalCharges"] = pd.to_numeric(ref["TotalCharges"], errors="coerce")

    rows = []
    for col in cur.columns:
        if col not in ref.columns:
            continue
        value = numeric_psi(ref[col], cur[col]) if col in NUMERIC else categorical_psi(ref[col], cur[col])
        rows.append((col, value, label(value)))

    report = pd.DataFrame(rows, columns=["feature", "psi", "status"]).sort_values("psi", ascending=False)
    print(f"Compared {len(cur)} logged predictions against {len(ref)} training rows\n")
    print(report.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    mean_proba = np.mean([r["churn_probability"] for r in records])
    train_rate = (ref["Churn"] == "Yes").mean()
    print(f"\nMean predicted churn probability: {mean_proba:.3f} (training churn rate: {train_rate:.3f})")

    if (report["status"] == "SIGNIFICANT").any():
        print("\nDRIFT DETECTED: consider retraining.")
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Check logged API traffic for drift against training data")
    p.add_argument("--log", default=os.path.join(PROJECT_ROOT, "logs", "predictions.jsonl"))
    p.add_argument("--reference", default=os.path.join(PROJECT_ROOT, "data", "OG", "Telco-Customer-Churn.csv"))
    p.add_argument("--min_rows", type=int, default=30)
    main(p.parse_args())
