import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def tune_model(X, y, n_trials: int = 20, scoring: str = "roc_auc"):
    """
    Tunes an XGBoost model using Optuna.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        n_trials (int): Number of Optuna trials.
        scoring (str): Threshold-independent metric to maximize; the decision
            threshold is chosen separately, so recall is not tuned here.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best Params:", study.best_params)
    return study.best_params
