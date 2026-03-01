# train.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score


DATA_PATH = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_OUTPUT_PATH = "serving_model.pkl"


def main():
    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape before cleaning: {df.shape}")

    # Drop identifier column
    df = df.drop(columns=["customerID"])

    # Fix datatype issues
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Remove rows with missing values
    df = df.dropna()

    print(f"Dataset shape after cleaning: {df.shape}")

    # ------------------------------------------------------------------
    # Prepare features and target
    # ------------------------------------------------------------------
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ------------------------------------------------------------------
    # Build preprocessing + model pipeline
    # ------------------------------------------------------------------
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    y_pred = pipeline.predict(X_test)

    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()