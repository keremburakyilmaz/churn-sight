import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

def load_processed_data(path="data/processed/"):
    X_train = pd.read_csv(f"{path}X_train.csv")
    X_test = pd.read_csv(f"{path}X_test.csv")
    y_train = pd.read_csv(f"{path}y_train.csv").squeeze()
    y_test = pd.read_csv(f"{path}y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

def train_baseline():
    X_train, X_test, y_train, y_test = load_processed_data()

    mlflow.set_experiment("churn-baseline")

    with mlflow.start_run(run_name = "lightgbm-baseline"):
        model = lgb.LGBMClassifier(random_state = 42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "accuracy": acc,
            "roc_auc": roc_auc
        })

        mlflow.lightgbm.log_model(model, artifact_path = "model")

        print(f"Baseline model trained: Accuracy={acc:.4f}, ROC AUC={roc_auc:.4f}")

train_baseline()
