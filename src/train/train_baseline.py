import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

def load_processed_data(path="data/processed/"):
    X_train = pd.read_csv(f"{path}X_train.csv")
    X_test = pd.read_csv(f"{path}X_test.csv")
    y_train = pd.read_csv(f"{path}y_train.csv").squeeze()
    y_test = pd.read_csv(f"{path}y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

# Train a baseline LightGBM model and log results to MLflow
def train_baseline(params):
    X_train, X_test, y_train, y_test = load_processed_data()
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    return model, y_pred, roc_auc, acc, y_proba