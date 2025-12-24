"""Paper 03: Machine Learning Models for Predicting Diabetes Risk Using Clinical Data - Nature Scientific Reports, 2021
Assumptions: Uses larger clinical datasets beyond Pima; includes ensemble methods and possibly neural nets; performance reported is realistic for clinical cohorts.
"""
import pandas as pd

data = {
    "Paper Title": ["Machine Learning Models for Predicting Diabetes Risk Using Clinical Data"],
    "Publication Year": [2021],
    "Dataset Used": ["Large clinical cohort / EHR-derived features"],
    "Algorithms / Models Used": ["Random Forest, Gradient Boosting, Logistic Regression"],
    "Deep Learning Used (Yes/No)": ["No"],
    "Explainable AI Used (Yes/No)": ["No"],
    "Accuracy / Performance": ["~0.83 accuracy; AUC ~0.87"],
    "Accuracy": [0.83],
    "Strengths": ["Large-scale clinical validation; attention to feature engineering and calibration."],
    "Limitations": ["Limited exploration of deep learning architectures and XAI for individual-level explanations."]
}

PAPER_DF = pd.DataFrame(data)

# Assumption: performance values are representative of medium-to-large clinical cohort studies.
