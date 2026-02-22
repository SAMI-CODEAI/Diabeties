"""
Combined Dataset Training Script
Trains an XGBoost model on the 150k merged dataset with hyperparameter tuning and XAI support.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, confusion_matrix
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'data/combined_diabetes_data.csv'
MODEL_PATH = 'models/xgb_model_combined.pkl'
SCALER_PATH = 'models/scaler_combined.pkl'
METRICS_PATH = 'models/model_metrics.txt'
TRAIN_SAMPLE_PATH = 'models/train_data_sample.csv'

def setup_directories():
    """Create necessary directories"""
    if not os.path.exists('models'):
        os.makedirs('models')

def load_and_preprocess_data():
    """Load data, handle duplicates, scale, and split"""
    print("="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    # Load dataset with python engine for better stability
    try:
        df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"Failed to load with python engine: {e}")
        # Fallback
        df = pd.read_csv(DATA_PATH, encoding='latin1')

    print(f"Loaded dataset: {df.shape}")
    
    # Drop metadata columns
    drop_cols = ['Dataset_Source', 'Match_Quality_Score']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Handle duplicates (logging them but keeping them as requested by user methodology for population representation)
    n_duplicates = df.duplicated().sum()
    print(f"Found {n_duplicates} duplicate rows (keeping them as valid population samples)")
    
    # Separate Features and Target
    X = df.drop('Diabetes', axis=1)
    y = df['Diabetes']
    
    feature_names = X.columns.tolist()
    print(f"Features ({len(feature_names)}): {feature_names}")
    
    # Create a unique signature for each Pima profile to prevent data leakage
    # We group by the original Pima features so that the same patient doesn't end up in both Train and Test
    pima_cols = ['Pima_Pregnancies', 'Pima_Glucose', 'Pima_BloodPressure', 
                 'Pima_SkinThickness', 'Pima_Insulin', 'Pima_DiabetesPedigreeFunction', 
                 'BMI', 'Age'] # BMI and Age in combined match Pima's
                 
    # Create string signature
    df['Pima_Signature'] = df[pima_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    
    unique_patients = df['Pima_Signature'].unique()
    print(f"Unique Pima patients identified: {len(unique_patients)} (Original Pima size ~768)")
    
    # Split signatures into Train/Test
    train_sigs, test_sigs = train_test_split(unique_patients, test_size=0.2, random_state=42)
    
    # Create masks
    train_mask = df['Pima_Signature'].isin(train_sigs)
    test_mask = df['Pima_Signature'].isin(test_sigs)
    
    X_train = df[train_mask].drop(['Diabetes', 'Pima_Signature'], axis=1)
    y_train = df[train_mask]['Diabetes']
    
    X_test = df[test_mask].drop(['Diabetes', 'Pima_Signature'], axis=1)
    y_test = df[test_mask]['Diabetes']
    
    # Clean up temporary column from df if needed (though we already extracted X/y)
    
    print(f"Training set: {X_train.shape} (Patients: {len(train_sigs)})")
    print(f"Test set: {X_test.shape} (Patients: {len(test_sigs)})")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    
    # Save a sample of training data for XAI (LIME/Anchors need this)
    # Using raw values (inverse transform later if needed) or scaled? 
    # Usually better to save scaled for consistency if model expects scaled
    # But for LIME interpretability, raw data is often preferred. 
    # We'll save the raw sample for now, app can scale it.
    train_sample = X_train.sample(n=5000, random_state=42)
    train_sample.to_csv(TRAIN_SAMPLE_PATH, index=False)
    print(f"Training sample saved to {TRAIN_SAMPLE_PATH}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def train_model(X_train, y_train):
    """Train XGBoost model with RandomizedSearchCV"""
    print("\n" + "="*80)
    print("MODEL TRAINING & HYPERPARAMETER TUNING")
    print("="*80)
    
    # Calculate scale_pos_weight for imbalanced classes
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Hyperparameter Grid
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9, 12],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'scale_pos_weight': [1, scale_pos_weight] # Try both balanced and standard
    }
    
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=30, # 30 iterations for thorough tuning
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=3),
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Starting RandomizedSearchCV (this may take a while)...")
    random_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {random_search.best_params_}")
    print(f"Best ROC-AUC: {random_search.best_score_:.4f}")
    
    best_model = random_search.best_estimator_
    
    # Save model
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return best_model

def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate model performance"""
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn) # Recall
    
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    # Feature Importance
    print("\nFeature Importance (Gain):")
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print(feat_imp.head(10))
    
    # Save metrics to file
    with open(METRICS_PATH, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"\nBest Params: {model.get_params()}")

def main():
    setup_directories()
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, feature_names)
    print("\nTraining workflow completed successfully!")

if __name__ == "__main__":
    main()
