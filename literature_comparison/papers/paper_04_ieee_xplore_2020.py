"""Paper 04: Comparative Study of Machine Learning Algorithms for Diabetes Prediction - IEEE Xplore, 2020
Assumptions: Strong comparative study; may present micro-averaged performance across folds.
"""
import pandas as pd

data = {
    "Paper Title": ["Comparative Study of Machine Learning Algorithms for Diabetes Prediction"],
    "Publication Year": [2020],
    "Dataset Used": ["Pima Indian Diabetes Dataset / Combined clinical datasets"],
    "Algorithms / Models Used": ["SVM, Random Forest, KNN, Logistic Regression"],
    "Deep Learning Used (Yes/No)": ["No"],
    "Explainable AI Used (Yes/No)": ["No"],
    "Accuracy / Performance": ["~0.79-0.85 depending on method; SVM/RF top performers"],
    "Accuracy": [0.82],
    "Strengths": ["Broad algorithm comparison and sensitivity analysis."],
    "Limitations": ["Often limited to small datasets; no model interpretability module."]
}

PAPER_DF = pd.DataFrame(data)

# Note: Accuracy aggregated to a representative value for visualization.
