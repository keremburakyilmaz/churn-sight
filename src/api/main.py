from fastapi import FastAPI
from src.api.schema import CustomerFeatures
import joblib
import numpy as np
import os

app = FastAPI()

# Load model
MODEL_PATH = os.path.join("models", "lightgbm_best_model.pkl")
model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running."}

@app.post("/predict")
def predict_churn(data: CustomerFeatures):
    # Convert input to ordered array
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

    input_data = np.array([[getattr(data, feat) for feat in feature_order]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }
