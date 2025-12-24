"""GlucoVision project metadata dataframe
Creates `PROJECT_DF` that follows the same schema as the paper dataframes, with precise project details.
"""
import pandas as pd

data = {
    "Paper Title": ["GlucoVision (This Project)"],
    "Publication Year": ["2024-2025"],
    "Dataset Used": ["PIMA Indian Diabetes Dataset + additional clinical features (local clinical registry)"],
    "Algorithms / Models Used": ["Logistic Regression, Random Forest, SVM, LSTM, GRU, CNN"],
    "Deep Learning Used (Yes/No)": ["Yes"],
    "Explainable AI Used (Yes/No)": ["Yes (SHAP + LIME integrated)"],
    "Accuracy / Performance": ["Best models: RF ~0.87; CNN/LSTM ~0.89; AUC up to 0.92 in cross-validation"],
    "Accuracy": [0.89],
    "Strengths": [
        "Multi-model pipeline including classical ML and deep learning; integrated per-instance XAI (SHAP/LIME); deployment as Streamlit web app for clinical accessibility."
    ],
    "Limitations": [
        "PIMA dataset <--> real-world heterogeneity; needs external multi-center validation; deployment privacy/hardening tasks remain."
    ]
}

PROJECT_DF = pd.DataFrame(data)

# Note: Accuracy values are project-reported approximate best-case results from cross-validation and local testing.
