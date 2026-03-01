# 🧠 Telco Customer Churn Prediction API

This project trains a Machine Learning model to predict customer churn  
and exposes it through a REST API using **FastAPI**.

The goal of this implementation is to demonstrate practical ML engineering skills, including:

- Data preprocessing
- Pipeline construction
- Model training & evaluation
- Model serialization
- REST API development
- Basic testing

---

## 📁 Project Structure

```
churn-api/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── serving_model.pkl
└── tests/
    └── test_api.py
```

### File Overview

- **train.py** → Data cleaning, preprocessing, model training, evaluation, and serialization  
- **app.py** → FastAPI application serving the trained model  
- **tests/test_api.py** → Basic API tests using pytest  
- **serving_model.pkl** → Serialized trained pipeline (generated after training)

---

## ⚙️ Environment Setup (Anaconda)

### 1️⃣ Create and activate environment

```bash
conda create -n churn-api python=3.10
conda activate churn-api
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

```bash
python train.py
```

This will:

- Clean and preprocess the dataset  
- Train a Random Forest classifier  
- Print evaluation metrics (Classification Report + F1 Score)  
- Save the trained pipeline as `serving_model.pkl`

---

## 🚀 Run the API

```bash
uvicorn app:app --reload
```

Open the interactive documentation:

```
http://127.0.0.1:8000/docs
```

You can test the `/predict` endpoint directly from Swagger UI.

---

## 📬 Example Request (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 45.3,
        "TotalCharges": 540.0
     }'
```

Example response:

```json
{
  "prediction": "No",
  "churn_probability": 0.16
}
```

---

## 🩺 Health Check

Verify the API is running:

```
http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

---

## 🧪 Run Tests

Basic API tests are implemented using `pytest`.

```bash
pytest
```

Tests validate:

- API startup
- `/health` endpoint
- `/predict` endpoint response structure

---

## 📊 Model Details

- **Algorithm:** Random Forest Classifier  
- **Preprocessing:**
  - OneHotEncoding for categorical variables
  - Numerical features passed through unchanged
- **Train/Test Split:** 80/20 (stratified)
- **Evaluation Metric:** F1 Score  
- **Model Serving:** Full sklearn pipeline serialized via `joblib`

---

## 🔐 Input Validation

The API uses Pydantic models to:

- Enforce required fields
- Validate numeric ranges (e.g., non-negative tenure and charges)
- Ensure proper request structure

Invalid inputs return automatic HTTP 422 responses.

---

## 🔮 Potential Production Improvements

If extended for production use, the following improvements could be considered:

- Model versioning and experiment tracking
- Automated hyperparameter tuning
- Structured logging and monitoring
- Docker containerization
- CI/CD integration
- Input schema versioning
- Performance benchmarking

---

## ✅ Summary

This implementation focuses on:

- Clarity  
- Reproducibility  
- Clean engineering practices  
- Practical ML system design  

It demonstrates an end-to-end workflow from data processing to deployment-ready API.