import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from train_optuna import run_optuna, load_processed_data

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iters):
            model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(model)
            dw = (1 / len(X)) * np.dot(X.T, (predictions - y))
            db = (1 / len(X)) * np.sum(predictions - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return (self._sigmoid(linear_model) >= 0.5).astype(int)

    def predict_proba(self, X):
        X = np.array(X)
        return self._sigmoid(np.dot(X, self.weights) + self.bias)

def train_logistic_regression(params):
    X_train, X_test, y_train, y_test = load_processed_data()
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return model, {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

def logistic_regression_objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 1.0, log=True),
        "n_iters": trial.suggest_int("n_iters", 500, 3000)
    }
    _, scores = train_logistic_regression(params)
    return scores["roc_auc"]

if __name__ == "__main__":
    run_optuna(logistic_regression_objective, train_logistic_regression, algo_name="logistic_regression", n_trials=5)
