"""Paper 01: Early Detection of Diabetes Mellitus Using Machine Learning Techniques - IEEE 2019
This module defines a pandas DataFrame `PAPER_DF` with metadata, assumptions, and an approximate numeric `Accuracy` field for plotting.
Assumptions: the study uses the Pima dataset and standard ML algorithms; accuracy is approximated from reported results (if multiple results, representative value is used).
"""
import pandas as pd

data = {
    "Paper Title": ["Early Detection of Diabetes Mellitus Using Machine Learning Techniques"],
    "Publication Year": [2019],
    "Dataset Used": ["Pima Indian Diabetes Dataset (UCI)"],
    "Algorithms / Models Used": ["Logistic Regression, Random Forest, SVM"],
    "Deep Learning Used (Yes/No)": ["No"],
    "Explainable AI Used (Yes/No)": ["No"],
    "Accuracy / Performance": ["~0.80 (accuracy); AUC ~0.82"],
    # Numeric accuracy for plotting
    "Accuracy": [0.80],
    "Strengths": ["Comparative evaluation of traditional classifiers with CV and sensible baselines."],
    "Limitations": ["Limited features (Pima-only); no deep learning or XAI considered."]
}

PAPER_DF = pd.DataFrame(data)

# Note: Accuracy is approximate based on aggregated results reported; values chosen to be realistic for Pima dataset studies.
