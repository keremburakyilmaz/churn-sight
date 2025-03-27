import json
from fastapi import FastAPI
import shap
import pandas as pd
from src.api.schema import CustomerFeatures
import joblib
import numpy as np
import os
from pydantic import BaseModel
from typing import List
from datetime import datetime

app = FastAPI()

# Load model
MODEL_PATH = os.path.join("models", "lightgbm_best_model.pkl")
model = joblib.load(MODEL_PATH)

X_train = pd.read_csv("data/processed/X_train.csv")

explainer = shap.TreeExplainer(model, data=X_train)

feature_order = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "gender_Male", "Partner_Yes", "Dependents_Yes",
    "PhoneService_Yes", "MultipleLines_No_phone_service", "MultipleLines_Yes",
    "InternetService_Fiber_optic", "InternetService_No",
    "OnlineSecurity_No_internet_service", "OnlineSecurity_Yes",
    "OnlineBackup_No_internet_service", "OnlineBackup_Yes",
    "DeviceProtection_No_internet_service", "DeviceProtection_Yes",
    "TechSupport_No_internet_service", "TechSupport_Yes",
    "StreamingTV_No_internet_service", "StreamingTV_Yes",
    "StreamingMovies_No_internet_service", "StreamingMovies_Yes",
    "Contract_One_year", "Contract_Two_year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit_card_automatic", "PaymentMethod_Electronic_check", "PaymentMethod_Mailed_check"
]

class CustomerBatch(BaseModel):
    customers: List[CustomerFeatures]

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running."}

def make_prediction(data: CustomerFeatures):
    # Helper function to do prediction
    input_data = np.array([[getattr(data, feat) for feat in feature_order]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return prediction, probability, input_data

@app.post("/predict")
def predict_churn(data: CustomerFeatures):
    # Predict churn
    prediction, probability, _ = make_prediction(data)
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }

@app.post("/explain")
def explain_prediction(data: CustomerFeatures):
    # Tell which features contributed the most to the prediction
    prediction, probability, input_data = make_prediction(data)
    shap_values = explainer.shap_values(input_data)[1]
    feature_impacts = list(zip(feature_order, shap_values[0]))
    top_features = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)[:3]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "top_features": [
            {"feature": feat, "impact": round(float(impact), 4)}
            for feat, impact in top_features
        ]
    }

@app.post("/batch_predict")
def batch_predict(batch: CustomerBatch):
    # Predict a batch of customers rather than just one
    results = []
    for customer in batch.customers:
        prediction, probability, _ = make_prediction(customer)
        results.append({
            "churn_prediction": int(prediction),
            "churn_probability": round(float(probability), 4)
        })
    return {"predictions": results}

@app.get("/model_info")
def get_model_info():
    with open("models/model_metadata.json") as f:
        metadata = json.load(f)

    return {
        "model_path": metadata["model_path"],
        "model_last_modified": metadata["trained_at"],
        "feature_count": len(feature_order),
        "features": feature_order,
        "metrics": {
            "accuracy": metadata["accuracy"],
            "roc_auc": metadata["roc_auc"]
        },
        "best_params": metadata["best_params"]
    }