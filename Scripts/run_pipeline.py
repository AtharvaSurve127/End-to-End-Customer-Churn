#!/usr/bin/env python3
"""
Training pipeline: load -> validate -> preprocess -> features -> tune (optional)
-> cross-validate -> pick threshold -> train -> evaluate -> save artifacts.

Usage:
    python Scripts/run_pipeline.py --input data/OG/Telco-Customer-Churn.csv [--tune]
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Allow imports from the src/ package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features, get_column_groups
from src.utils.validate_data import validate_telco_data
from src.models.evaluate import (
    probability_metrics, baseline_scores, choose_threshold, calibration_figure,
)
from src.models.train import DEFAULT_PARAMS, out_of_fold_proba, train_final
from src.models.tune import tune_model

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


def main(args):
    mlflow.set_tracking_uri(args.mlflow_uri or f"sqlite:///{PROJECT_ROOT.replace(os.sep, '/')}/mlflow.db")
    mlflow.set_experiment(args.experiment)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    with mlflow.start_run() as run:
        mlflow.log_param("test_size", args.test_size)

        # --- Load and validate ---
        print("Loading data...")
        df = load_data(args.input)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        print("Validating data quality with Great Expectations...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))
        if not is_valid:
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        print("Data validation passed.")

        # --- Preprocess and build features ---
        df = preprocess_data(df, target_col=args.target)
        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found in data")

        processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)

        binary_cols, multi_cols = get_column_groups(df, args.target)
        df_enc = build_features(df, target_col=args.target)
        X = df_enc.drop(columns=[args.target])
        y = df_enc[args.target]

        # Everything serving needs to reproduce the encoding
        schema = {
            "feature_columns": list(X.columns),
            "binary_cols": binary_cols,
            "multi_cols": multi_cols,
            "target": args.target,
        }
        with open(os.path.join(ARTIFACTS_DIR, "feature_schema.json"), "w") as f:
            json.dump(schema, f, indent=2)
        mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "feature_schema.json"))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=42
        )
        print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

        # --- Hyperparameters ---
        params = dict(DEFAULT_PARAMS)
        if args.tune:
            print(f"Tuning with Optuna ({args.n_trials} trials, metric: roc_auc)...")
            params.update(tune_model(X_train, y_train, n_trials=args.n_trials))
        for k, v in params.items():
            mlflow.log_param(f"xgb_{k}", v)
        mlflow.log_param("tuned", args.tune)

        # --- Cross-validation and threshold (training data only) ---
        print(f"{args.cv_folds}-fold cross-validation...")
        oof = out_of_fold_proba(params, X_train, y_train, folds=args.cv_folds)
        for name, val in probability_metrics(y_train, oof).items():
            mlflow.log_metric(f"cv_{name}", val)
            print(f"   CV {name}: {val:.3f}")

        if args.threshold is None:
            threshold, _ = choose_threshold(y_train, oof, args.cost_fp, args.cost_fn)
            print(f"Cost-optimal threshold: {threshold:.2f} "
                  f"(cost_fp={args.cost_fp}, cost_fn={args.cost_fn})")
        else:
            threshold = args.threshold
        mlflow.log_params({"threshold": threshold, "cost_fp": args.cost_fp, "cost_fn": args.cost_fn})

        # --- Train final model ---
        t0 = time.time()
        model, explain_model = train_final(params, X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"Model trained in {train_time:.2f}s")

        # --- Evaluate on the held-out test set ---
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "test_cost": int(((y_pred == 1) & (y_test == 0)).sum()) * args.cost_fp
                         + int(((y_pred == 0) & (y_test == 1)).sum()) * args.cost_fn,
            **probability_metrics(y_test, proba),
        }
        for name, val in metrics.items():
            mlflow.log_metric(name, val)
        print(f"Test (threshold {threshold:.2f}): precision {metrics['precision']:.3f} | "
              f"recall {metrics['recall']:.3f} | F1 {metrics['f1']:.3f}")
        print(f"   ROC AUC {metrics['roc_auc']:.3f} | PR AUC {metrics['avg_precision']:.3f} | "
              f"Brier {metrics['brier']:.3f} | cost {metrics['test_cost']}")

        mlflow.log_figure(calibration_figure(y_test, proba), "calibration.png")

        print("Baselines (test set):")
        for name, sc in baseline_scores(X_train, y_train, X_test, y_test).items():
            print(f"   {name}: ROC AUC {sc['roc_auc']:.3f} | Brier {sc['brier']:.3f}")
            for k, v in sc.items():
                mlflow.log_metric(f"baseline_{name}_{k}", v)

        # --- Save serving artifacts ---
        joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.joblib"))
        joblib.dump(explain_model, os.path.join(ARTIFACTS_DIR, "explain_model.pkl"))
        with open(os.path.join(ARTIFACTS_DIR, "threshold.json"), "w") as f:
            json.dump({"threshold": threshold, "cost_fp": args.cost_fp, "cost_fn": args.cost_fn}, f)
        with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json"), "w") as f:
            json.dump({
                "run_id": run.info.run_id,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tuned": args.tune,
                "params": params,
                "test_metrics": {k: float(v) for k, v in metrics.items()},
            }, f, indent=2)
        for fname in ["model.joblib", "explain_model.pkl", "threshold.json", "model_metadata.json"]:
            mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, fname))
        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"Artifacts saved to {ARTIFACTS_DIR}")

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/OG/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=None,
                   help="fixed decision threshold; default: pick by expected cost")
    p.add_argument("--cost_fp", type=float, default=1.0,
                   help="cost of a retention offer sent to a customer who would stay")
    p.add_argument("--cost_fn", type=float, default=5.0,
                   help="cost of missing a customer who churns")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--tune", action="store_true", help="run Optuna before training")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="override MLflow tracking URI, else sqlite:///<project>/mlflow.db")
    main(p.parse_args())
