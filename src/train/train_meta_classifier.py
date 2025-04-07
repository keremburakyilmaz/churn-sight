import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from train_optuna import load_processed_data

from train_logistic_regression import LogisticRegression
from train_mlp import MLPClassifier
from train_random_forest import RandomForestClassifier
from train_xgboost import XGBoostClassifier
from train_decision_tree import DecisionTreeClassifier
from train_gaussian_naive_bayes import GaussianNaiveBayes
from train_lgbm import train_lgbm


class MetaClassifier:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_predictions = None

    def fit(self, X, y):
        base_outputs = []
        for model in self.base_models:
            model.fit(X, y)
            base_outputs.append(model.predict_proba(X))
        self.base_predictions = np.column_stack(base_outputs)
        self.meta_model.fit(self.base_predictions, y)

    def predict(self, X):
        base_outputs = []
        for model in self.base_models:
            base_outputs.append(model.predict_proba(X))
        stacked_input = np.column_stack(base_outputs)
        return self.meta_model.predict(stacked_input)

    def predict_proba(self, X):
        base_outputs = []
        for model in self.base_models:
            base_outputs.append(model.predict_proba(X))
        stacked_input = np.column_stack(base_outputs)
        return self.meta_model.predict_proba(stacked_input)


def train_meta_classifier():
    X_train, X_test, y_train, y_test = load_processed_data()

    lgbm_model, _, _, _, _ = train_lgbm({
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 100
    })

    class LightGBMWrapper:
        def __init__(self, model):
            self.model = model

        def fit(self, X, y):
            pass

        def predict_proba(self, X):
            return self.model.predict_proba(X)

    base_models = [
        LogisticRegression(lr=0.01, n_iters=1000),
        MLPClassifier(input_size=X_train.shape[1], hidden_size=64, output_size=1, lr=0.01, epochs=50),
        RandomForestClassifier(n_estimators=10, max_depth=5),
        XGBoostClassifier(n_estimators=5, learning_rate=0.1),
        DecisionTreeClassifier(max_depth=5, min_samples_split=2),
        GaussianNaiveBayes(),
        LightGBMWrapper(lgbm_model)
    ]

    meta_model = LogisticRegression(lr=0.01, n_iters=1000)
    meta_clf = MetaClassifier(base_models, meta_model)
    meta_clf.fit(X_train, y_train)

    y_pred = meta_clf.predict(X_test)
    y_proba = meta_clf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Meta Classifier (Custom Logistic Meta) - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    return meta_clf


if __name__ == "__main__":
    train_meta_classifier()
