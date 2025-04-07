import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from train_optuna import run_optuna, load_processed_data

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(np.array(X), np.array(y))

    def _grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return TreeNode(value=np.mean(y))

        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            return TreeNode(value=np.mean(y))

        left_mask = X[:, best_feat] < best_thresh
        right_mask = ~left_mask

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return TreeNode(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y):
        best_gain = -float("inf")
        split_idx, split_thresh = None, None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                gain = self._gain(X[:, feature], y, t)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = t
        return split_idx, split_thresh

    def _gain(self, feature_column, y, threshold):
        left_mask = feature_column < threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return -float("inf")

        y_left, y_right = y[left_mask], y[right_mask]
        gain = np.var(y) - (len(y_left) / len(y)) * np.var(y_left) - (len(y_right) / len(y)) * np.var(y_right)
        return gain

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in np.array(X)])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class XGBoostClassifier:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_pred = np.zeros(len(y))

        for _ in range(self.n_estimators):
            residual = y - self._sigmoid(y_pred)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residual)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def predict_proba(self, X):
        X = np.array(X)
        pred = np.zeros(X.shape[0])
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return self._sigmoid(pred)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

def train_xgboost(params):
    X_train, X_test, y_train, y_test = load_processed_data()
    model = XGBoostClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return model, {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

def xgboost_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
    }
    _, scores = train_xgboost(params)
    return scores["roc_auc"]

if __name__ == "__main__":
    run_optuna(xgboost_objective, train_xgboost, algo_name="xgboost", n_trials=5)
