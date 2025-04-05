import numpy as np
import joblib
from sklearn.metrics import roc_auc_score

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)  
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return (self.sigmoid(linear) >= 0.5).astype(int)

    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        probs = self.sigmoid(linear)
        return np.stack([1 - probs, probs], axis=1)

def evaluate_model(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, proba)

def save_model(model, path="models/logistic_regression_best_model.pkl"):
    joblib.dump(model, path)

def load_model(path="models/logistic_regression_best_model.pkl"):
    return joblib.load(path)
