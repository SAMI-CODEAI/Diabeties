"""Paper 10: Interpretable Machine Learning Models in Healthcare Diagnostics - ACM Computing Surveys, 2021
Assumptions: Survey paper covering ML, DL, and XAI methods in healthcare diagnostics; not dataset-specific.
"""
import pandas as pd

data = {
    "Paper Title": ["Interpretable Machine Learning Models in Healthcare Diagnostics"],
    "Publication Year": [2021],
    "Dataset Used": ["Survey across multiple clinical datasets (N/A specific)"],
    "Algorithms / Models Used": ["Survey: LR, SVM, RF, DL, XAI methods (LIME/SHAP)"],
    "Deep Learning Used (Yes/No)": ["Yes (survey covers DL literature)"],
    "Explainable AI Used (Yes/No)": ["Yes (LIME/SHAP and interpretable models discussed)"],
    "Accuracy / Performance": ["N/A (survey); reported ranges across studies: 0.75-0.90"],
    "Accuracy": [0.82],
    "Strengths": ["Comprehensive literature synthesis and taxonomy of interpretability methods."],
    "Limitations": ["Survey-level conclusions; does not present new empirical results."]
}

PAPER_DF = pd.DataFrame(data)

# Remark: Surveys are valuable for trends but may not provide single numeric benchmarks.
