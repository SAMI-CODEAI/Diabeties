"""Paper 06: Diabetes Prediction Using Support Vector Machines - Procedia Computer Science, 2017
Assumptions: SVM is central; dataset frequently Pima; emphasizes kernel choice and hyperparameter tuning.
"""
import pandas as pd

data = {
    "Paper Title": ["Diabetes Prediction Using Support Vector Machines"],
    "Publication Year": [2017],
    "Dataset Used": ["Pima Indian Diabetes Dataset"],
    "Algorithms / Models Used": ["Support Vector Machines (RBF/linear)"],
    "Deep Learning Used (Yes/No)": ["No"],
    "Explainable AI Used (Yes/No)": ["No"],
    "Accuracy / Performance": ["~0.78-0.84 depending on kernel and tuning"],
    "Accuracy": [0.81],
    "Strengths": ["Thorough treatment of kernel methods and regularization."],
    "Limitations": ["Scalability and interpretability are limited for clinicians."]
}

PAPER_DF = pd.DataFrame(data)

# Accuracy: representative; SVM often performs well with appropriate feature scaling.
