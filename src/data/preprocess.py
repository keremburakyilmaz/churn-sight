import pandas as pd
from sklearn.model_selection import train_test_split

def save_processed_data(X_train, X_test, y_train, y_test, path="data/processed/"):
    X_train.to_csv(f"{path}X_train.csv", index=False)
    X_test.to_csv(f"{path}X_test.csv", index=False)
    y_train.to_csv(f"{path}y_train.csv", index=False)
    y_test.to_csv(f"{path}y_test.csv", index=False)

def load_data(filepath="data/raw/telco_churn.csv"):
    df = pd.read_csv(filepath)

    # Clean whitespaces
    df.columns = df.columns.str.replace(" ", "_")

    # Drop customerID, not useful for predictions
    df.drop('customerID', axis=1, inplace=True)

    # Target encoding, needed for classification
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Convert total charges to numeric (handle missing)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    # Train-test split
    # All columns except the target is X
    X = df.drop('Churn', axis=1)

    # Target column is y
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    save_processed_data(X_train, X_test, y_train, y_test)

load_data()
