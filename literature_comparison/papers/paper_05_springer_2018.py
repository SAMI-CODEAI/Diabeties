"""Paper 05: Random Forest-Based Predictive Model for Type 2 Diabetes - Springer, 2018
Assumptions: Paper emphasizes RF as ideal for tabular clinical data and reports strong feature importance analysis.
"""
import pandas as pd

data = {
    "Paper Title": ["Random Forest-Based Predictive Model for Type 2 Diabetes"],
    "Publication Year": [2018],
    "Dataset Used": ["Clinical study cohorts / Pima"],
    "Algorithms / Models Used": ["Random Forest (primary), Decision Trees"],
    "Deep Learning Used (Yes/No)": ["No"],
    "Explainable AI Used (Yes/No)": ["Partial (feature importance)"],
    "Accuracy / Performance": ["~0.82 accuracy; strong variable importance metrics"],
    "Accuracy": [0.82],
    "Strengths": ["Interpretable variable importance; robust handling of tabular data."],
    "Limitations": ["Feature importance is global; lacks per-instance explanations and DL exploration."]
}

PAPER_DF = pd.DataFrame(data)

# Remarks: 'Explainable AI' is interpreted as feature importance methods here (global explainability).
