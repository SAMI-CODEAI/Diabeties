# train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

DATA_PATH = "data/pima_diabetes.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

EXPECTED_FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

def load_and_prepare(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns.astype(str)]
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")
    df = df.dropna(subset=EXPECTED_FEATURES + ["Outcome"])
    X = df[EXPECTED_FEATURES]
    y = df["Outcome"].astype(int)
    return X, y

def main():
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    probas = model.predict_proba(X_test_scaled)[:, 1]

    print("Classification report:")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probas))

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    print("Saved scaler and model to", MODEL_DIR)

if __name__ == "__main__":
    main()
