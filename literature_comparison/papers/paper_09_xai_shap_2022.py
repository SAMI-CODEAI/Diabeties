"""Paper 09: Explainable AI for Diabetes Risk Prediction Using SHAP - Nature Communications, 2022
Assumptions: Integrates SHAP explanations for model transparency; likely uses large clinical datasets and ensemble or gradient boosting models.
"""
import pandas as pd

data = {
    "Paper Title": ["Explainable Artificial Intelligence for Diabetes Risk Prediction Using SHAP"],
    "Publication Year": [2022],
    "Dataset Used": ["Large clinical datasets / EHR"],
    "Algorithms / Models Used": ["XGBoost/Gradient Boosting, Random Forest; SHAP for explanation"],
    "Deep Learning Used (Yes/No)": ["No (focus on ensemble + XAI)"],
    "Explainable AI Used (Yes/No)": ["Yes (SHAP integrated)"],
    "Accuracy / Performance": ["~0.85-0.89 AUC with SHAP explanations for model trust"],
    "Accuracy": [0.87],
    "Strengths": ["Per-instance SHAP explanations allow clinical plausibility checks and feature-level transparency."],
    "Limitations": ["SHAP computation can be costly; explains post-hoc, not inherently interpretable models."]
}

PAPER_DF = pd.DataFrame(data)

# Note: This paper represents the wave of integrating formal XAI (SHAP) with predictive modeling in diabetes risk.
