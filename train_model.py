# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
from utils import EXPECTED_FEATURES

DATA_PATH = "data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_prepare(path=DATA_PATH):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns.astype(str)]
    
    # Verify columns
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    
    # Target
    if "Diabetes_binary" not in df.columns:
        raise ValueError("Target column 'Diabetes_binary' not found.")
        
    X = df[EXPECTED_FEATURES]
    y = df["Diabetes_binary"].astype(int)
    return X, y

def main():
    X, y = load_and_prepare()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save a raw sample of training data for DiCE/Anchors/LIME in app
    # We'll save a small effective sample to keep it light, or full if feasible. 
    # The app needs it for "training_data" reference in explainers.
    print("Saving training data sample for XAI...")
    # Setup dataframe with target
    train_sample = X_train.copy()
    train_sample["Diabetes_binary"] = y_train
    train_sample.sample(n=min(len(train_sample), 5000), random_state=42).to_csv(
        os.path.join(MODEL_DIR, "train_data_sample.csv"), index=False
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Starting RandomizedSearchCV for XGBoost...")
    # Hyperparameter tuning
    params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'gamma': [0, 0.1, 0.2],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    
    search = RandomizedSearchCV(
        xgb, 
        param_distributions=params, 
        n_iter=10, 
        scoring='roc_auc', 
        n_jobs=-1, 
        cv=3, 
        verbose=1,
        random_state=42
    )
    
    search.fit(X_train_scaled, y_train)
    
    print(f"Best params: {search.best_params_}")
    print(f"Best AUC: {search.best_score_}")
    
    model = search.best_estimator_
    
    # Evaluate
    preds = model.predict(X_test_scaled)
    probas = model.predict_proba(X_test_scaled)[:, 1]
    
    print("Classification report:")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probas))
    
    # Save
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    print("Saved scaler and model to", MODEL_DIR)

if __name__ == "__main__":
    main()
