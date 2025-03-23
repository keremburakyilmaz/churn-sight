import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath="data/raw/telco_churn.csv"):
    df = pd.read_csv(filepath)

    # Drop customerID, not useful for predictions
    df.drop('customerID', axis=1, inplace=True)

    # Target encoding, needed for classification
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Convert total charges to numeric (handle missing)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    # Train-test split

    # All columns except the target is X
    X = df.drop('Churn', axis=1)

    # Target column is y
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
