import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from train_optuna import run_optuna, load_processed_data


class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        if (depth >= self.max_depth or num_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        best_feat, best_thresh = self._best_split(X, y, n_features)
        if best_feat is None:
            return DecisionTreeNode(value=self._most_common_label(y))

        left_indices = X[:, best_feat] < best_thresh
        right_indices = X[:, best_feat] >= best_thresh

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return DecisionTreeNode(value=self._most_common_label(y))

        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return DecisionTreeNode(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                gain = self._information_gain(y, X[:, feature], t)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = t

        return split_idx, split_thresh

    def _information_gain(self, y, feature_values, threshold):
        parent_entropy = self._entropy(y)

        left = y[feature_values < threshold]
        right = y[feature_values >= threshold]

        if len(left) == 0 or len(right) == 0:
            return 0

        n = len(y)
        weighted_entropy = (len(left) / n) * self._entropy(left) + (len(right) / n) * self._entropy(right)
        return parent_entropy - weighted_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p + 1e-10) for p in ps if p > 0])

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def predict_proba(self, X):
        preds = self.predict(X)
        return np.column_stack((1 - preds, preds))

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


def train_decision_tree(params):
    X_train, X_test, y_train, y_test = load_processed_data()
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return model, {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def decision_tree_objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20)
    }
    _, scores = train_decision_tree(params)
    return scores["roc_auc"]


if __name__ == "__main__":
    run_optuna(decision_tree_objective, train_decision_tree, algo_name="decision_tree", n_trials=5)
