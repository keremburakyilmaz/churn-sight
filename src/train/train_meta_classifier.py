import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from train_optuna import load_processed_data
from train_logistic_regression import LogisticRegression
from train_decision_tree import DecisionTreeClassifier
from train_random_forest import RandomForestClassifier
from train_xgboost import XGBoostClassifier
from train_lgbm import train_lgbm
from train_mlp import MLPClassifier
from train_gaussian_naive_bayes import GaussianNaiveBayes

def train_meta_classifier():
    X_train, X_test, y_train, y_test = load_processed_data()

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_train_pred = dt.predict_proba(X_train)[:, 1]
    dt_test_pred = dt.predict_proba(X_test)[:, 1]

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_train_pred = rf.predict_proba(X_train)[:, 1]
    rf_test_pred = rf.predict_proba(X_test)[:, 1]

    xgb = XGBoostClassifier()
    xgb.fit(X_train, y_train)
    xgb_train_pred = xgb.predict_proba(X_train)
    xgb_test_pred = xgb.predict_proba(X_test)

    mlp = MLPClassifier(input_size=X_train.shape[1])
    mlp.fit(X_train, y_train)
    mlp_train_pred = mlp.predict_proba(X_train)
    mlp_test_pred = mlp.predict_proba(X_test)

    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    nb_train_pred = nb.predict_proba(X_train)
    nb_test_pred = nb.predict_proba(X_test)

    lgbm_model, _ = train_lgbm({
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0
    })
    lgb_train_pred = lgbm_model.predict_proba(X_train)[:, 1]
    lgb_test_pred = lgbm_model.predict_proba(X_test)[:, 1]

    print("Stacking predictions for meta-classifier...")
    meta_X_train = np.column_stack([
        dt_train_pred, rf_train_pred, xgb_train_pred, mlp_train_pred, nb_train_pred, lgb_train_pred
    ])
    meta_X_test = np.column_stack([
        dt_test_pred, rf_test_pred, xgb_test_pred, mlp_test_pred, nb_test_pred, lgb_test_pred
    ])

    meta_model = LogisticRegression(lr=0.1, n_iters=1000)
    meta_model.fit(meta_X_train, y_train)

    meta_pred = meta_model.predict(meta_X_test)
    meta_proba = meta_model.predict_proba(meta_X_test)

    acc = accuracy_score(y_test, meta_pred)
    auc = roc_auc_score(y_test, meta_proba)

    print(f"\nMeta-classifier evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")

if __name__ == "__main__":
    train_meta_classifier()
