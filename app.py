# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI
app = FastAPI()

# Load trained pipeline once at startup
model = joblib.load("serving_model.pkl")


# Define input schema
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(data: CustomerInput):
    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    churn_label = "Yes" if prediction == 1 else "No"

    return {
        "prediction": churn_label,
        "churn_probability": float(probability)
    }

@app.get("/health")
def health():
    return {"status": "ok"}