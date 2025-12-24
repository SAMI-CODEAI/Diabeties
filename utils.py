# utils.py
import pandas as pd
import numpy as np
import os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

EXPECTED_FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

# PIMA Dataset Medians for Imputation
MEDIANS = {
    "SkinThickness": 23.00,
    "Insulin": 30.50,
    "DiabetesPedigreeFunction": 0.3725,
    "Pregnancies": 0 # Default to 0 if not provided (e.g. Male)
}

# directories for saving explanation images
SHAP_DIR = os.path.join("static", "shap_images")
LIME_DIR = os.path.join("static", "lime_images")
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(LIME_DIR, exist_ok=True)

def normalize_columns(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns.astype(str)]
    return df

def validate_and_prepare_df(df: pd.DataFrame):
    df = normalize_columns(df)
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Expected: {EXPECTED_FEATURES}")
    df = df[EXPECTED_FEATURES].copy()
    for c in EXPECTED_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[EXPECTED_FEATURES].isnull().any(axis=None):
        raise ValueError("Found NaNs after converting data. Ensure all values are numeric and present.")
    return df

def single_input_to_df(form_dict):
    row = {}
    
    # helper to safely get float
    def get_val(key):
        v = form_dict.get(key)
        if v is None or str(v).strip() == "":
            return None
        try:
            return float(v)
        except:
            return None

    # 1. Handle Basic Inputs
    row["Age"] = get_val("Age")
    row["Glucose"] = get_val("Glucose")
    row["BloodPressure"] = get_val("BloodPressure")
    row["Pregnancies"] = get_val("Pregnancies") # Might be None

    # 2. Handle BMI or Height/Weight
    bmi_val = get_val("BMI")
    if bmi_val is not None:
        row["BMI"] = bmi_val
    else:
        # Try Calc BMI
        h_cm = get_val("Height")
        w_kg = get_val("Weight")
        if h_cm and w_kg:
            # BMI = kg / m^2
            h_m = h_cm / 100.0
            row["BMI"] = round(w_kg / (h_m * h_m), 1)
        else:
            row["BMI"] = None # Will error or need default? Let's leave None to catch later if needed.

    # 3. Handle Advanced/Technical Inputs (Impute if Missing)
    for feat in ["SkinThickness", "Insulin", "DiabetesPedigreeFunction"]:
        val = get_val(feat)
        if val is not None:
            row[feat] = val
        else:
            # IMPUTE MEDIAN
            row[feat] = MEDIANS.get(feat, 0)

    # 4. Handle Pregnancies Default
    if row["Pregnancies"] is None:
        row["Pregnancies"] = 0 # Default

    return pd.DataFrame([row])

def generate_care_plan(row_dict, shap_values, proba_percent):
    """
    Generate a natural language explanation and action plan.
    row_dict: dict of feature values for this user
    shap_values: array of shap values (impact) for features
    proba_percent: risk score (0-100)
    """
    plan = {
        "summary": "",
        "key_factors": [],
        "actions": []
    }
    
    # 1. High Level Summary
    if proba_percent < 30:
        plan["summary"] = "Your results indicate a low risk profile currently. However, maintaining a healthy lifestyle is crucial."
    elif proba_percent < 70:
        plan["summary"] = "Your results indicate an elevated risk. It is advisable to address specific lifestyle factors."
    else:
        plan["summary"] = "Your results indicate a high risk profile. Immediate consultation with a healthcare professional is strongly recommended."

    # 2. Identify heavy hitters from SHAP
    # feature names are EXPECTED_FEATURES.
    # We want features that pushed risk UP (positive shap).
    
    # Map feature index to name
    feat_map = {i: n for i, n in enumerate(EXPECTED_FEATURES)}
    
    # Sort by impact (highest positive first)
    # shap_values is typically (1, n_features) or (n_features,)
    if len(shap_values.shape) == 2:
        vals = shap_values[0]
    else:
        vals = shap_values
        
    # Zip and sort
    impacts = []
    for i, val in enumerate(vals):
        impacts.append((feat_map[i], val))
    
    impacts.sort(key=lambda x: x[1], reverse=True)
    
    # Top 3 'bad' factors
    top_risks = [x for x in impacts if x[1] > 0][:3]
    
    for feat, score in top_risks:
        val = row_dict.get(feat, "?")
        if feat == "Glucose":
            msg = f"Your Glucose level ({val} mg/dL) is a primary contributor."
            action = "Reduce refined sugars and carbohydrates. Monitor fasting sugar regularly."
        elif feat == "BMI":
            msg = f"Your BMI ({val}) contributes to the risk."
            action = "Aim for a gradual weight loss of 5-10% through diet and moderate exercise."
        elif feat == "Age":
            msg = f"Age ({val}) is a natural risk factor."
            action = "Regular screenings are essential as we age. Focus on what you can control: diet and activity."
        elif feat == "BloodPressure":
            msg = f"Blood Pressure ({val} mmHg) is influencing the score."
            action = "Monitor BP. Reduce salt intake and manage stress."
        elif feat == "Insulin":
            msg = "Insulin levels contributed to the prediction."
            action = "Discuss Insulin resistance with your endocrinologist."
        elif feat == "DiabetesPedigreeFunction":
            msg = "Family history increases genetic predisposition."
            action = "Since genetics are fixed, be extra vigilant with lifestyle choices."
        else:
            msg = f"{feat} ({val}) is a contributing factor."
            action = "Discuss this specific metric with your doctor."
        
        plan["key_factors"].append({"factor": msg, "action": action})

    if not plan["key_factors"]:
        plan["key_factors"].append({"factor": "No single factor stands out.", "action": "Maintain balanced habits."})

    return plan

def generate_shap_plot(model, scaler, X_raw, feature_names=EXPECTED_FEATURES):
    """
    model: trained sklearn-like model (trained on scaled inputs)
    scaler: fitted scaler (StandardScaler)
    X_raw: 2D numpy array or DataFrame with original raw feature values (1 x n_features)
    returns: relative path to saved PNG (e.g., static/shap_images/...)
    """
    # Convert X_raw to numpy 2D
    if isinstance(X_raw, pd.DataFrame):
        X_raw_vals = X_raw.values
    else:
        X_raw_vals = np.array(X_raw)
    # Scale (model trained on scaled)
    X_scaled = scaler.transform(X_raw_vals)
    # Use TreeExplainer on model
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_scaled)
    except Exception:
        # Fallback newer API
        explainer = shap.TreeExplainer(model)
        shap_exp = explainer(X_scaled)
        shap_vals = shap_exp.values

    # Convert to array of shape (n_samples, n_features)
    shap_arr = np.array(shap_vals)
    # For single sample:
    idx = 0
    vals = shap_arr[idx]
    abs_vals = np.abs(vals)
    order = np.argsort(abs_vals)[::-1]
    top_n = min(len(feature_names), 8)
    top_idx = order[:top_n]
    top_feats = [feature_names[i] for i in top_idx]
    top_vals = vals[top_idx]

    plt.figure(figsize=(6,4))
    # horizontal bar with sign
    y_pos = range(len(top_feats))
    plt.barh(list(reversed(top_feats)), list(reversed(top_vals)))
    plt.xlabel("SHAP value (impact on model output)")
    plt.title("SHAP - Top feature impacts")
    plt.tight_layout()

    fname = f"shap_{int(time.time()*1000)}.png"
    rel_path = os.path.join("static", "shap_images", fname)
    plt.savefig(rel_path, bbox_inches="tight")
    plt.close()
    return rel_path.replace("\\", "/")

def generate_lime_plot(model, scaler, X_raw, training_df_raw, feature_names=EXPECTED_FEATURES, class_names=["NotDiabetic","Diabetic"]):
    """
    model: trained model
    scaler: fitted scaler
    X_raw: single-row DataFrame or 2D-array in raw units
    training_df_raw: DataFrame of raw training samples (unscaled) to build the LIME explainer
    returns: relative path to saved PNG (e.g., static/lime_images/...)
    """
    # Prepare training data for explainer (LIME expects raw data if our predict_fn accepts raw)
    train_np = np.array(training_df_raw[feature_names].values)

    # define prediction function that accepts raw numpy array (n_samples x n_features)
    def predict_fn(raw_array):
        # raw_array -> scale -> model.predict_proba
        scaled = scaler.transform(raw_array)
        probs = model.predict_proba(scaled)
        return probs

    explainer = LimeTabularExplainer(
        training_data=train_np,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True
    )

    # X_raw as 1D array for explain_instance
    if isinstance(X_raw, pd.DataFrame):
        instance = X_raw.iloc[0].values
    else:
        instance = np.array(X_raw).reshape(-1)

    exp = explainer.explain_instance(instance, predict_fn, num_features=min(8, len(feature_names)))

    # Plot LIME explanation as horizontal bar
    # exp.as_list() returns list of (feature, contribution)
    feature_contribs = exp.as_list()
    feats = [f for f, _ in feature_contribs]
    vals = [v for _, v in feature_contribs]

    # Build bar plot
    plt.figure(figsize=(6,4))
    # LIME values can be positive/negative; horizontal bar
    plt.barh(list(reversed(feats)), list(reversed(vals)))
    plt.xlabel("LIME feature contribution (to predicted class)")
    plt.title("LIME - Local explanation")
    plt.tight_layout()

    fname = f"lime_{int(time.time()*1000)}.png"
    rel_path = os.path.join("static", "lime_images", fname)
    plt.savefig(rel_path, bbox_inches="tight")
    plt.close()
    return rel_path.replace("\\", "/")




