import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from train_optuna import run_optuna, load_processed_data
from train_decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _feature_subset(self, X):
        n_features = X.shape[1]
        if self.max_features == 'sqrt':
            size = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            size = int(np.log2(n_features))
        else:
            size = n_features
        return np.random.choice(n_features, size=size, replace=False)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.trees = []

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            feature_indices = self._feature_subset(X_sample)
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        X = np.array(X)
        tree_preds = np.array([
            tree.predict(X[:, features]) for tree, features in self.trees
        ])
        return np.round(np.mean(tree_preds, axis=0)).astype(int)

    def predict_proba(self, X):
        X = np.array(X)
        tree_probas = np.array([
            tree.predict_proba(X[:, features])[:, 1] for tree, features in self.trees
        ])
        mean_proba = np.mean(tree_probas, axis=0)
        return np.column_stack((1 - mean_proba, mean_proba))


def train_random_forest(params):
    X_train, X_test, y_train, y_test = load_processed_data()
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return model, {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def random_forest_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 5, 50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    }
    _, scores = train_random_forest(params)
    return scores["roc_auc"]


if __name__ == "__main__":
    run_optuna(random_forest_objective, train_random_forest, algo_name="random_forest", n_trials=5)
