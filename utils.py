# utils.py
import pandas as pd
import numpy as np
import os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# Optional XAI libraries
try:
    import dice_ml
except ImportError:
    dice_ml = None

try:
    from alibi.explainers import AnchorTabular
except ImportError:
    AnchorTabular = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

EXPECTED_FEATURES = [
    "Pima_Pregnancies", "Pima_Glucose", "Pima_BloodPressure", 
    "Pima_SkinThickness", "Pima_Insulin", "Pima_DiabetesPedigreeFunction",
    "BMI", "Age", 
    "CDC_BMI", "CDC_Age_Category",
    "CDC_HighBP", "CDC_HighChol", "CDC_CholCheck", "CDC_Smoker",
    "CDC_Stroke", "CDC_HeartDiseaseorAttack", "CDC_PhysActivity", 
    "CDC_Fruits", "CDC_Veggies", "CDC_HvyAlcoholConsump", 
    "CDC_AnyHealthcare", "CDC_NoDocbcCost", "CDC_GenHlth",
    "CDC_MentHlth", "CDC_PhysHlth", "CDC_DiffWalk", 
    "CDC_Sex", "CDC_Education", "CDC_Income"
]

# directories for saving explanation images
SHAP_DIR = os.path.join("static", "shap_images")
LIME_DIR = os.path.join("static", "lime_images")
DICE_DIR = os.path.join("static", "dice_images") # For html output if needed
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(LIME_DIR, exist_ok=True)
os.makedirs(DICE_DIR, exist_ok=True)

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
    
    # helper to safely get int/float
    def get_val(key, default=0):
        v = form_dict.get(key)
        if v is None or str(v).strip() == "":
            return default
        try:
            return float(v)
        except:
            return default

    # 1. Map Clinical Features (Pima)
    row["Pima_Pregnancies"] = get_val("Pregnancies", 0)
    row["Pima_Glucose"] = get_val("Glucose", 120)       # Default healthy-ish if missing
    row["Pima_BloodPressure"] = get_val("BloodPressure", 70) # Default healthy
    row["Pima_SkinThickness"] = get_val("SkinThickness", 20) # Median
    row["Pima_Insulin"] = get_val("Insulin", 80)        # Median
    row["Pima_DiabetesPedigreeFunction"] = get_val("DiabetesPedigreeFunction", 0.3)

    # 2. Map Common Features (Age/BMI)
    age_raw = get_val("Age", 30) # Default 30 years
    bmi_raw = get_val("BMI", 25.0)

    row["BMI"] = bmi_raw
    row["Age"] = age_raw
    
    # CDC Duplicates
    row["CDC_BMI"] = bmi_raw
    
    # CDC Age Category Calculation (Approximate inverse of mapping)
    # 18-24=1, 25-29=2, 30-34=3, 35-39=4, 40-44=5, 45-49=6, 50-54=7, 
    # 55-59=8, 60-64=9, 65-69=10, 70-74=11, 75-79=12, 80+=13
    if age_raw < 25: cat = 1
    elif age_raw < 30: cat = 2
    elif age_raw < 35: cat = 3
    elif age_raw < 40: cat = 4
    elif age_raw < 45: cat = 5
    elif age_raw < 50: cat = 6
    elif age_raw < 55: cat = 7
    elif age_raw < 60: cat = 8
    elif age_raw < 65: cat = 9
    elif age_raw < 70: cat = 10
    elif age_raw < 75: cat = 11
    elif age_raw < 80: cat = 12
    else: cat = 13
    row["CDC_Age_Category"] = cat

    # 3. Map CDC Lifestyle Features
    # Note: Form inputs match CDC suffixes usually e.g. "HighBP"
    row["CDC_HighBP"] = get_val("HighBP", 0)
    row["CDC_HighChol"] = get_val("HighChol", 0)
    row["CDC_CholCheck"] = get_val("CholCheck", 1)
    row["CDC_Smoker"] = get_val("Smoker", 0)
    row["CDC_Stroke"] = get_val("Stroke", 0)
    row["CDC_HeartDiseaseorAttack"] = get_val("HeartDiseaseorAttack", 0)
    row["CDC_PhysActivity"] = get_val("PhysActivity", 1)
    row["CDC_Fruits"] = get_val("Fruits", 1)
    row["CDC_Veggies"] = get_val("Veggies", 1)
    row["CDC_HvyAlcoholConsump"] = get_val("HvyAlcoholConsump", 0)
    row["CDC_AnyHealthcare"] = get_val("AnyHealthcare", 1)
    row["CDC_NoDocbcCost"] = get_val("NoDocbcCost", 0)
    row["CDC_GenHlth"] = get_val("GenHlth", 2) # Very Good default
    row["CDC_MentHlth"] = get_val("MentHlth", 0)
    row["CDC_PhysHlth"] = get_val("PhysHlth", 0)
    row["CDC_DiffWalk"] = get_val("DiffWalk", 0)
    row["CDC_Sex"] = get_val("Sex", 0)
    row["CDC_Education"] = get_val("Education", 5) # Some college
    row["CDC_Income"] = get_val("Income", 6) # Middle income default
    
    return pd.DataFrame([row])

def generate_care_plan(row_dict, shap_values, proba_percent):
    """
    Generate a formatted Clinical Decision Support report with detailed explanations.
    """
    plan = {
        "summary": "",
        "key_factors": [],
        "actions": []
    }
    
    # 1. Humanize the Risk
    if proba_percent < 30:
        plan["summary"] = "Low Risk Profile"
    elif proba_percent < 50:
        plan["summary"] = "Moderate Risk - Monitor Closely"
    elif proba_percent < 70:
        plan["summary"] = "Borderline High - Action Required"
    else:
        plan["summary"] = "High Risk - Clinical Attention Recommended"

    # 2. Identify top contributing factors from SHAP
    feat_map = {i: n for i, n in enumerate(EXPECTED_FEATURES)}
    
    # Check shape of shap_values
    if isinstance(shap_values, list):
         vals = shap_values[0]
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
        vals = shap_values[0]
    else:
        vals = shap_values
        
    impacts = []
    for i, val in enumerate(vals):
        impacts.append((feat_map[i], val))
    
    # Sort by impact (highest positive first = increased risk)
    impacts.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Rich Explanations Dictionary (Why & How)
    EXPLANATIONS = {
        "Pima_Glucose": {
            "why": "Blood Sugar Level",
            "how": "Elevated fasting glucose indicates your body is struggling to process sugar, a direct marker of diabetes risk."
        },
        "Pima_BloodPressure": {
            "why": "Cardiovascular Strain",
            "how": "Higher diastolic pressure suggests your heart is working harder between beats, often linked to insulin resistance."
        },
        "Pima_Insulin": {
            "why": "Insulin Resistance",
            "how": "High insulin levels often mean your body is producing more to compensate for cells not responding, a precursor to burnout."
        },
        "Pima_SkinThickness": {
             "why": "Body Fat Distribution",
             "how": "Triceps skinfold thickness estimates subcutaneous body fat, which correlates with overall metabolic health."
        },
        "Pima_DiabetesPedigreeFunction": {
             "why": "Genetic Predisposition",
             "how": "A higher score indicates a strong family history, meaning your genetic baseline risk is elevated."
        },
        "Pima_Pregnancies": {
             "why": "Gestational Stress",
             "how": "Multiple pregnancies can place repeated metabolic stress on the body, sometimes unmasking latent insulin issues."
        },
        "CDC_HighBP": {
            "why": "Systemic Strain",
            "how": "Chronic high blood pressure damages blood vessels, forcing the heart to work harder and increasing the risk of metabolic complications."
        },
        "CDC_HighChol": {
            "why": "Arterial Plaque Build-up",
            "how": "Elevated cholesterol leads to plaque accumulation in arteries, restricting blood flow and raising cardiovascular and diabetes risk."
        },
        "BMI": {
            "why": "Adipose Tissue Impact",
            "how": "Higher body mass, particularly visceral fat, promotes inflammation and insulin resistance, the core drivers of Type 2 diabetes."
        },
        "CDC_Smoker": {
            "why": "Oxidative Stress",
            "how": "Smoking introduces toxins that damage cells, causing oxidative stress and directly impairing insulin sensitivity."
        },
        "CDC_PhysActivity": {
            "why": "Sedentary Lifestyle",
            "how": "Lack of regular muscle engagement reduces glucose uptake from the blood, leading to higher sustained blood sugar levels."
        },
        "CDC_Fruits": {
            "why": "Micronutrient Deficiency",
            "how": "A diet low in whole fruits lacks essential fiber and antioxidants that help regulate blood sugar absorption."
        },
        "CDC_Veggies": {
            "why": "Low Dietary Fiber",
            "how": "Vegetables are a primary source of fiber, which slows digestion and prevents blood sugar spikes."
        },
        "CDC_HvyAlcoholConsump": {
            "why": "Liver Stress",
            "how": "Excessive alcohol consumption places strain on the liver, disrupting its ability to regulate blood glucose effectively."
        },
        "CDC_GenHlth": {
            "why": "Overall Health Status",
            "how": "Your self-reported poor health often correlates with underlying undiagnosed inflammation or chronic stress."
        },
        "CDC_MentHlth": {
            "why": "Psychological Stress",
            "how": "Frequent mental distress can elevate cortisol levels, a hormone that naturally increases blood sugar."
        },
        "CDC_PhysHlth": {
            "why": "Physical Limitation",
            "how": "Frequent physical illness limits your ability to maintain an active, calorie-burning lifestyle."
        },
        "CDC_DiffWalk": {
            "why": "Mobility Restriction",
            "how": "Difficulty walking severely limits physical activity options, contributing to weight gain and muscle atrophy."
        },
        "CDC_Sex": {
            "why": "Biological Factors",
            "how": "Biological differences can influence fat distribution and hormonal baselines affecting risk calculation."
        },
        "Age": {
            "why": "Metabolic Aging",
            "how": "As we age, pancreatic function naturally declines and cells become more resistant to insulin."
        },
        "CDC_Education": {
            "why": "Socioeconomic Correlation",
            "how": "Statistical models often find correlations between education levels and access to healthcare or health literacy."
        },
         "CDC_HeartDiseaseorAttack": {
            "why": "Comorbidity",
            "how": "Prior cardiovascular events indicate an existing compromise in vascular health, which is closely linked to diabetes pathology."
        },
        "CDC_Stroke": {
            "why": "Vascular History",
            "how": "A history of stroke suggests significant vascular risk factors are already present."
        },
         "CDC_CholCheck": {
            "why": "Preventative Gap",
            "how": "Irregular cholesterol screening may mean missed opportunities to manage lipid levels early."
        }
    }
    
    ACTIONABLE = [
        "BMI", "CDC_Smoker", "CDC_PhysActivity", "CDC_Fruits", "CDC_Veggies", 
        "CDC_HvyAlcoholConsump", "CDC_HighBP", "CDC_HighChol", "Pima_Glucose", "Pima_BloodPressure", "Pima_Insulin"
    ]
    
    # 4. Build Action Plan (Modifiable only)
    top_risks = [x for x in impacts if x[1] > 0]
    
    for feat, score in top_risks:
        val = row_dict.get(feat, 0)
        action_text = None
        
        # Skip if not actionable
        if feat not in ACTIONABLE:
            continue
            
        if feat == "CDC_HighBP" and val == 1:
            action_text = "• Blood Pressure: Management is critical. A DASH diet (low sodium) and stress reduction are proven first-line defenses."
        elif feat == "CDC_HighChol" and val == 1:
            action_text = "• Cholesterol: Swap saturated fats (red meat, butter) for healthy fats (nuts, olive oil) to improve lipid profiles."
        elif feat == "BMI":
            if val > 25:
                 action_text = f"• Weight: Your BMI is {val}. Even a modest 5% weight loss improves insulin sensitivity dramatically."
        elif feat == "CDC_Smoker" and val == 1:
            action_text = "• Smoking: Quitting is the most effective way to restore your body's cardiovascular resilience."
        elif feat == "CDC_PhysActivity" and val == 0:
            action_text = "• Activity: Aim for 150 mins/week of brisk walking. Muscles are the main consumers of blood sugar."
        elif feat == "CDC_Fruits" and val == 0:
            action_text = "• Nutrition: Add one serving of whole fruit (berries, apple) to your daily routine for fiber."
        elif feat == "CDC_Veggies" and val == 0:
             action_text = "• Nutrition: Aim to fill half your plate with non-starchy vegetables at every meal."
        elif feat == "CDC_HvyAlcoholConsump" and val == 1:
            action_text = "• Alcohol: Reducing intake relieves metabolic stress on the liver."
        elif feat == "Pima_Glucose" and val > 140:
             action_text = f"• Blood Sugar: Your glucose ({val}) is elevated. Consult your doctor for an A1C test."
        elif feat == "Pima_Insulin" and val > 160:
             action_text = "• Insulin: High insulin suggests resistance. Intermittent fasting or carb reduction can help sensitive cells."

        if action_text:
            plan["actions"].append(action_text)
            
    plan["actions"] = plan["actions"][:4]
    if not plan["actions"]:
        plan["actions"].append("Prioritize maintaining your current healthy preventive habits.")

    # 5. Key Factors Analysis (The Why & How)
    # Mention top features regardless of modifiability, but exclude Income
    for feat, score in top_risks[:5]: # Check top 5 to ensure we get 4 after filtering
        # Skip Income - don't show it in the UI
        if feat == "Income":
            continue
            
        val = row_dict.get(feat, "?")
        exp = EXPLANATIONS.get(feat, {"why": "Risk Factor", "how": "This factor statistically increases the probability of a diagnosis."})
        
        label_class = "Modifiable" if feat in ACTIONABLE else "Non-Modifiable"
        
        plan["key_factors"].append({
            "factor": feat,
            "value": val,
            "label": label_class,
            "why": exp["why"],
            "how": exp["how"]
        })
        
        # Stop once we have 4 factors (excluding Income)
        if len(plan["key_factors"]) >= 4:
            break

    return plan

# Remove DiCE function logic (Stub)
def generate_dice_bcf(*args, **kwargs):
    return []

def generate_anchor_rule(model, X_raw, X_train_numpy, feature_names=EXPECTED_FEATURES, class_names=["Healthy", "Diabetic"]):
    """
    Generate Anchors rule. Safe against shape mismatches.
    """
    try:
        if AnchorTabular is None:
             return ["Anchors library not installed."]

        # Ensure X_train_numpy matches feature_names shape
        # expecting (n_samples, n_features)
        if len(X_train_numpy.shape) == 2:
            cols_in_train = X_train_numpy.shape[1]
            if cols_in_train > len(feature_names):
                # Assume target is included (likely last or similar), but safest to trust features if columns known
                # But X_train_numpy is often just array. 
                # If it's 22 vs 21, likely last column is target.
                X_train_clean = X_train_numpy[:, :len(feature_names)]
            else:
                X_train_clean = X_train_numpy
        else:
             X_train_clean = X_train_numpy

        predict_fn = lambda x: model.predict(x)
        explainer = AnchorTabular(predict_fn, feature_names)
        explainer.fit(X_train_clean)
        
        instance = X_raw.values[0]
        # ensure instance shape
        explanation = explainer.explain(instance, threshold=0.95)
        
        return explanation.anchor
    except Exception as e:
        print(f"Anchors Error: {e}")
        return ["Could not generate rule."]

def generate_shap_plot(model, scaler, X_raw, feature_names=EXPECTED_FEATURES):
    """
    Generate SHAP summary plot for a single instance.
    """
    try:
        # 1. Scale input
        if isinstance(X_raw, pd.DataFrame):
            X_vals = X_raw.values
        else:
            X_vals = np.array(X_raw)
        
        X_scaled = scaler.transform(X_vals)
        
        # 2. Create Explainer (TreeExplainer for XGBoost)
        # Note: In production, you might want to load a pre-computed explainer
        explainer = shap.TreeExplainer(model)
        
        # 3. Get Shap Values
        shap_values = explainer.shap_values(X_scaled)
        
        # Handle different SHAP output formats (list for multiclass, array for binary)
        if isinstance(shap_values, list):
             vals = shap_values[0]
             if len(shap_values) > 1:
                 # IF binary, index 1 is usually the positive class
                 vals = shap_values[1]
        else:
             vals = shap_values

        # 4. Filter top features for plotting
        if len(vals.shape) == 2:
            vals = vals[0]
            
        df_shap = pd.DataFrame({
            'feature': feature_names,
            'shap_value': vals
        })
        # Add absolute value for sorting
        df_shap['abs_val'] = df_shap['shap_value'].abs()
        df_shap = df_shap.sort_values('abs_val', ascending=False).head(10)
        
        # 5. Plot
        plt.figure(figsize=(8, 5))
        # Horizontal bar chart
        # Color: Red for positive (risk), Blue for negative (protective)
        colors = ['#ff4d4d' if x > 0 else '#2ecc71' for x in df_shap['shap_value']]
        
        plt.barh(df_shap['feature'], df_shap['shap_value'], color=colors)
        plt.xlabel("SHAP Value (Impact on Model Output)")
        plt.title("Feature Impact for this Patient")
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.gca().invert_yaxis() # Highest impact on top
        plt.tight_layout()
        
        fname = f"shap_{int(time.time()*1000)}.png"
        rel_path = os.path.join("static", "shap_images", fname)
        plt.savefig(rel_path, bbox_inches="tight")
        plt.close()
        
        return f"static/shap_images/{fname}"
        
    except Exception as e:
        print(f"SHAP Error: {e}")
        import traceback
        traceback.print_exc()
        return ""


def generate_lime_plot(model, scaler, X_raw, training_df_raw, feature_names=EXPECTED_FEATURES, class_names=["NotDiabetic","Diabetic"]):
    """
    Generate LIME output. Safe against target column presence.
    """
    try:
        # Fix: Ensure training data only has expected features
        # Select only the features we need
        train_filtered = training_df_raw[feature_names].copy()
        train_np = train_filtered.values
    except Exception as e:
        print(f"LIME Data Prep Error: {e}")
        return ""

    # define prediction function
    def predict_fn(raw_array):
        # raw_array -> scale -> model.predict_proba
        scaled = scaler.transform(raw_array)
        probs = model.predict_proba(scaled)
        return probs

    try:
        if LimeTabularExplainer is None:
            print("LIME not installed.")
            return ""

        explainer = LimeTabularExplainer(
            training_data=train_np,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification",
            discretize_continuous=True
        )
        
        # Get instance as numpy array (1D)
        # Handle X_raw whether it is DataFrame or numpy array
        if isinstance(X_raw, pd.DataFrame):
            instance = X_raw.iloc[0].values
        else:
             instance = np.array(X_raw).reshape(-1)
        
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=min(8, len(feature_names))
        )
        
        # Save plot
        fname = f"lime_exp_{np.random.randint(10000)}.png"
        save_path = os.path.join("static", "lime_images", fname)
        
        # LIME plot to file
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        
        return f"static/lime_images/{fname}"
    except Exception as e:
        print(f"LIME Generation Error: {e}")
        return ""


# ===== Dataset Analysis Functions =====

DATASET_DIR = os.path.join("static", "dataset_analysis")
os.makedirs(DATASET_DIR, exist_ok=True)

def analyze_dataset_features(model, scaler, df_clean, predictions, probabilities):
    """
    Analyze feature importance for a dataset of predictions.
    Returns a list of dicts with feature names and importance percentages.
    """
    try:
        # Get feature importances from the model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Fallback: use mean absolute SHAP values across dataset
            import shap
            X_scaled = scaler.transform(df_clean.values)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            importances = np.abs(shap_values).mean(axis=0)
        
        # Normalize to percentages
        total = sum(importances)
        importance_pct = [(imp / total * 100) for imp in importances]
        
        # Create list of dicts
        feature_importance = []
        for i, feat in enumerate(EXPECTED_FEATURES):
            feature_importance.append({
                'feature': feat,
                'importance': round(importance_pct[i], 2)
            })
        
        # Sort by importance descending
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        return feature_importance
    except Exception as e:
        print(f"Error analyzing features: {e}")
        return []


def generate_feature_importance_plot(feature_importance, top_n=8):
    """
    Generate a bar plot of feature importance.
    Returns relative path to saved image.
    """
    try:
        # Filter out Income from the list
        filtered_importance = [f for f in feature_importance if f['feature'] != 'Income']
        
        # Take top N features from filtered list
        top_features = filtered_importance[:top_n]
        features = [f['feature'] for f in top_features]
        importances = [f['importance'] for f in top_features]
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        y_pos = range(len(features))
        
        plt.barh(list(reversed(features)), list(reversed(importances)), color=list(reversed(colors)))
        plt.xlabel('Importance (%)', fontsize=12, fontweight='bold')
        plt.title('Feature Importance - Major Effecting Factors', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fname = f"feature_importance_{int(time.time()*1000)}.png"
        rel_path = os.path.join("static", "dataset_analysis", fname)
        plt.savefig(rel_path, bbox_inches="tight", dpi=150)
        plt.close()
        return "/" + rel_path.replace("\\", "/")
    except Exception as e:
        print(f"Error generating feature plot: {e}")
        return None


def generate_distribution_plot(predictions, probabilities):
    """
    Generate visualization showing prediction distribution.
    Returns relative path to saved image.
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Prediction counts
        counts = np.bincount(predictions)
        labels = ['Not Diabetic', 'Diabetic']
        colors = ['#28a745', '#dc3545']
        axes[0].bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
        axes[0].set_title('Prediction Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(counts):
            axes[0].text(i, v + max(counts)*0.02, str(v), ha='center', fontweight='bold')
        
        # Plot 2: Probability distribution
        axes[1].hist(probabilities, bins=20, color='#007bff', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Predicted Probability (Diabetic)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1].set_title('Probability Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        fname = f"distribution_{int(time.time()*1000)}.png"
        rel_path = os.path.join("static", "dataset_analysis", fname)
        plt.savefig(rel_path, bbox_inches="tight", dpi=150)
        plt.close()
        return "/" + rel_path.replace("\\", "/")
    except Exception as e:
        print(f"Error generating distribution plot: {e}")
        return None


def calculate_dataset_statistics(df_clean):
    """
    Calculate statistics for each feature in the dataset.
    Returns dict of dicts with mean, median, std, min, max.
    """
    stats = {}
    for col in df_clean.columns:
        stats[col] = {
            'mean': float(df_clean[col].mean()),
            'median': float(df_clean[col].median()),
            'std': float(df_clean[col].std()),
            'min': float(df_clean[col].min()),
            'max': float(df_clean[col].max())
        }
    return stats


def save_predictions_csv(df_original, predictions, probabilities):
    """
    Save predictions to a CSV file.
    Returns the filename (not full path).
    """
    try:
        # Add predictions to the dataframe
        df_result = df_original.copy()
        df_result['Prediction'] = ['Diabetic' if p == 1 else 'Not Diabetic' for p in predictions]
        df_result['Probability_Diabetic'] = [round(prob * 100, 2) for prob in probabilities]
        
        fname = f"predictions_{int(time.time()*1000)}.csv"
        rel_path = os.path.join("static", "dataset_analysis", fname)
        df_result.to_csv(rel_path, index=False)
        
        return fname
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return None
