"""Paper 07: Deep Learning Approaches for Diabetes Prediction Using EHR - Springer Nature, 2020
Assumptions: Examines LSTM, CNN and dense networks on time-series EHR or derived features; reports stronger performance compared to classic ML on large datasets.
"""
import pandas as pd

data = {
    "Paper Title": ["Deep Learning Approaches for Diabetes Prediction Using Electronic Health Records"],
    "Publication Year": [2020],
    "Dataset Used": ["EHR-derived longitudinal datasets"],
    "Algorithms / Models Used": ["LSTM, CNN, Dense Neural Networks"],
    "Deep Learning Used (Yes/No)": ["Yes"],
    "Explainable AI Used (Yes/No)": ["Limited / No"],
    "Accuracy / Performance": ["~0.86 accuracy; AUC ~0.90 for well-tuned DL models"],
    "Accuracy": [0.86],
    "Strengths": ["Captures temporal patterns in EHR; improved performance on large data."],
    "Limitations": ["Challenge in interpretability; requires substantial data and compute."]
}

PAPER_DF = pd.DataFrame(data)

# Note: Deep learning shows gains on longitudinal EHR data; XAI is often not deeply integrated in such studies.
