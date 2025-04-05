import optuna
import pandas as pd
from train_logistic_regression import LogisticRegression, evaluate_model, save_model

# Load processed data
def load_processed_data(path="data/processed/"):
    X_train = pd.read_csv(f"{path}X_train.csv")
    X_test = pd.read_csv(f"{path}X_test.csv")
    y_train = pd.read_csv(f"{path}y_train.csv").squeeze()
    y_test = pd.read_csv(f"{path}y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_processed_data()

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1.0, log=True)
    n_iters = trial.suggest_int("n_iters", 500, 2000)

    model = LogisticRegression(lr=lr, n_iters=n_iters)
    model.fit(X_train, y_train)
    score = evaluate_model(model, X_test, y_test)
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best trial:")
print(study.best_trial)

# Train final model
best_params = study.best_trial.params
final_model = LogisticRegression(**best_params)
final_model.fit(X_train, y_train)
save_model(final_model)
