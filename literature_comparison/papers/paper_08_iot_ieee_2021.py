"""Paper 08: IoT and Machine Learning-Based System for Monitoring Diabetes - IEEE Sensors Journal, 2021
Assumptions: Integrates sensor data pipelines with ML models for continuous monitoring and predictive alerts.
"""
import pandas as pd

data = {
    "Paper Title": ["IoT and Machine Learning-Based System for Monitoring Diabetes"],
    "Publication Year": [2021],
    "Dataset Used": ["Sensor streams + clinical records; research demo datasets"],
    "Algorithms / Models Used": ["Random Forest, SVM, lightweight Neural Nets"],
    "Deep Learning Used (Yes/No)": ["Yes (lightweight CNN/NN for sensor features)"],
    "Explainable AI Used (Yes/No)": ["No"],
    "Accuracy / Performance": ["~0.80-0.88 depending on sensor modalities"],
    "Accuracy": [0.84],
    "Strengths": ["Real-time monitoring pipeline; emphasis on reliability and edge inferencing."],
    "Limitations": ["Data heterogeneity and privacy concerns; limited interpretability."]
}

PAPER_DF = pd.DataFrame(data)

# Comment: IoT papers emphasize system engineering and real-time performance rather than maximal offline accuracy.
