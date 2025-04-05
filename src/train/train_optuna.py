import optuna
import pandas as pd
from datetime import datetime
import mlflow
import joblib
import json

def load_processed_data(path="data/processed/"):
    X_train = pd.read_csv(f"{path}X_train.csv")
    X_test = pd.read_csv(f"{path}X_test.csv")
    y_train = pd.read_csv(f"{path}y_train.csv").squeeze()
    y_test = pd.read_csv(f"{path}y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

def run_optuna(objective_fn, train_fn, algo_name, n_trials=30, mlflow_exp="churn-optuna-tuning"):
    mlflow.set_experiment(mlflow_exp)
    study = optuna.create_study(direction="maximize")
    
    with mlflow.start_run(run_name=f"{algo_name}-optuna-search"):
        study.optimize(objective_fn, n_trials=n_trials)
        best_params = study.best_params
        print(f"Best params for {algo_name}: {best_params}")

        model, score_dict = train_fn(best_params)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_path = f"models/{algo_name}_best_model.pkl"
        joblib.dump(model, model_path)

        metadata = {
            "model_path": model_path,
            "trained_at": timestamp,
            **score_dict,
            "best_params": best_params
        }

        with open(f"models/{algo_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        mlflow.log_params(best_params)
        mlflow.log_metrics(score_dict)
        mlflow.log_artifact(model_path, artifact_path="model_files")

        print(f"{algo_name} model training complete and logged.")
