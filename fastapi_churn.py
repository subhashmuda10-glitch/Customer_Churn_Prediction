from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="Telco Churn Prediction API (FastAPI)")

# ----- 1. Request schema -----
class Customer(BaseModel):
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

# ----- 2. Load model -----
with open("models/best_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# ----- 3. Health / root -----
@app.get("/")
def read_root():
    return {"message": "FastAPI Telco Churn API is live"}

# ----- 4. Predict endpoint -----
@app.post("/predict")
def predict_churn(customer: Customer):
    data = customer.dict()
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    will_churn = prob > 0.4

    return {
        "churn_probability": round(float(prob), 4),
        "will_churn": bool(will_churn),
        "risk_level": "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.3 else "LOW"
    }
