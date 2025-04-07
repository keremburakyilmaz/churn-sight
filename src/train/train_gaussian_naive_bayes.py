import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from train_optuna import run_optuna, load_processed_data


class GaussianNaiveBayes:
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + 1e-9
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _pdf(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_instance(self, x):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(np.log(self._pdf(c, x)))
            posteriors.append(prior + likelihood)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_instance(x) for x in X])

    def predict_proba(self, X):
        X = np.array(X)
        probs = []
        for x in X:
            class_probs = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self._pdf(c, x)))
                class_probs.append(prior + likelihood)
            class_probs = np.exp(class_probs - np.max(class_probs))
            probs.append(class_probs / np.sum(class_probs))
        return np.array(probs)


def train_gaussian_naive_bayes(params=None):
    X_train, X_test, y_train, y_test = load_processed_data()
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return model, {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def gaussian_nb_objective(trial):
    _, scores = train_gaussian_naive_bayes()
    return scores["roc_auc"]


if __name__ == "__main__":
    run_optuna(gaussian_nb_objective, train_gaussian_naive_bayes, algo_name="gaussian_naive_bayes", n_trials=1)
