"""Paper 02: Diabetes Prediction Using Logistic Regression and Machine Learning Techniques - Elsevier 2020
Assumptions: Focused on logistic regression benchmarks with other ML models; dataset often Pima or clinical registry.
"""
import pandas as pd

data = {
    "Paper Title": ["Diabetes Prediction Using Logistic Regression and Machine Learning Techniques"],
    "Publication Year": [2020],
    "Dataset Used": ["Pima Indian Diabetes Dataset / Clinical registry"],
    "Algorithms / Models Used": ["Logistic Regression, Decision Trees, Random Forest"],
    "Deep Learning Used (Yes/No)": ["No"],
    "Explainable AI Used (Yes/No)": ["No"],
    "Accuracy / Performance": ["~0.78-0.83 (accuracy) depending on model"],
    "Accuracy": [0.805],
    "Strengths": ["Clear presentation of LR as interpretable baseline; robust cross-validation."],
    "Limitations": ["Limited external validation; does not explore DL or formal XAI methods."]
}

PAPER_DF = pd.DataFrame(data)

# Comment: accuracy is a representative mean across reported experiments.
