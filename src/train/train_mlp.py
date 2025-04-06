import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from train_optuna import run_optuna, load_processed_data


class MLPClassifier:
    def __init__(self, input_size, hidden_size=32, lr=0.01, n_iters=1000, activation="relu"):
        self.lr = lr
        self.n_iters = n_iters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self._init_weights()

    def _init_weights(self):
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def _activate(self, x):
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        raise ValueError(f"Unknown activation: {self.activation}")

    def _activate_derivative(self, x):
        if self.activation == "relu":
            return (x > 0).astype(float)
        elif self.activation == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return sig * (1 - sig)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        for _ in range(self.n_iters):
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self._activate(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self._sigmoid(Z2)

            dZ2 = A2 - y
            dW2 = np.dot(A1.T, dZ2) / len(X)
            db2 = np.sum(dZ2, axis=0, keepdims=True) / len(X)

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self._activate_derivative(Z1)
            dW1 = np.dot(X.T, dZ1) / len(X)
            db1 = np.sum(dZ1, axis=0, keepdims=True) / len(X)

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict_proba(self, X):
        X = np.array(X)
        A1 = self._activate(np.dot(X, self.W1) + self.b1)
        A2 = self._sigmoid(np.dot(A1, self.W2) + self.b2)
        return A2.flatten()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def train_mlp(params):
    X_train, X_test, y_train, y_test = load_processed_data()
    input_size = X_train.shape[1]
    model = MLPClassifier(input_size=input_size, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return model, {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def mlp_objective(trial):
    params = {
        "hidden_size": trial.suggest_int("hidden_size", 16, 128),
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "n_iters": trial.suggest_int("n_iters", 500, 3000),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
    }
    _, scores = train_mlp(params)
    return scores["roc_auc"]


if __name__ == "__main__":
    run_optuna(mlp_objective, train_mlp, algo_name="mlp", n_trials=30)
