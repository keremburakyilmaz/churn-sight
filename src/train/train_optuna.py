import optuna
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from optuna.integration.mlflow import MLflowCallback
from train_baseline import train_baseline

# Load processed data
def load_processed_data(path="data/processed/"):
    X_train = pd.read_csv(f"{path}X_train.csv")
    X_test = pd.read_csv(f"{path}X_test.csv")
    y_train = pd.read_csv(f"{path}y_train.csv").squeeze()
    y_test = pd.read_csv(f"{path}y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

# Objective function for Optuna
def objective(trial):
    # Define search space
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    # Train LightGBM model
    _, _, roc_auc, _, _ = train_baseline(params)

    return roc_auc

def train_with_optuna():
    mlflow.set_experiment("churn-optuna-tuning")

    # Optuna study
    study = optuna.create_study(direction="maximize")
    mlflow_callback = MLflowCallback(tracking_uri = mlflow.get_tracking_uri(), metric_name = "roc_auc")

    # Run optimization
    study.optimize(objective, n_trials=20, callbacks=[mlflow_callback])

    # Get best model params
    best_params = study.best_params
    print(f"Best params: {best_params}")

    # Save best model
    best_model, _, _, _, _ = train_baseline(best_params)

    # Log best model
    with mlflow.start_run(run_name="best-optuna-model"):
        mlflow.log_params(best_params)
        mlflow.lightgbm.log_model(best_model, "best_model")


train_with_optuna()
