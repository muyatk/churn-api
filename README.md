# Telco Customer Churn Prediction API

This project trains a Machine Learning model to predict customer churn  
and exposes it through a REST API using FastAPI.

The purpose of this implementation is to demonstrate practical ML engineering skills, including:

- Data preprocessing
- Pipeline construction
- Model training & evaluation
- Model serialization
- REST API deployment

---

## 📁 Project Structure

```
.
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

- **train.py** → Data cleaning, model training, evaluation, and model serialization  
- **app.py** → FastAPI application serving the trained model  
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

Open in your browser:

```
http://127.0.0.1:8000/docs
```

Use the interactive Swagger UI to test the `/predict` endpoint.

---

## 🩺 Health Check

To verify the API is running:

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

## 📊 Model Details

- **Algorithm:** Random Forest Classifier  
- **Preprocessing:**
  - OneHotEncoding for categorical variables
  - Numerical features passed through unchanged
- **Train/Test Split:** 80/20 (stratified)
- **Evaluation Metric:** F1 Score  

---

## 🔮 Possible Production Improvements

If deployed in production, potential improvements include:

- Model versioning
- Automated hyperparameter tuning
- Logging & monitoring
- Docker containerization
- CI/CD integration

---

## ✅ Summary

This implementation focuses on clarity, reproducibility, and clean engineering practices.