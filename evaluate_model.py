# evaluate_model.py
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score
)
from utils import EXPECTED_FEATURES

# Paths
DATA_PATH = "data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
MODEL_DIR = "models"
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data_and_model():
    """Load the dataset and trained model."""
    print("Loading data and model...")
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns.astype(str)]
    
    X = df[EXPECTED_FEATURES]
    y = df["Diabetes_binary"].astype(int)
    
    # Split (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load model and scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    return X_test, X_test_scaled, y_test, model

def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall (Sensitivity)': recall_score(y_true, y_pred),
        'Specificity': recall_score(y_true, y_pred, pos_label=0),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'Average Precision': average_precision_score(y_true, y_proba),
        'Matthews Correlation Coefficient': matthews_corrcoef(y_true, y_pred),
        'Cohen Kappa': cohen_kappa_score(y_true, y_pred)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(y_true, y_proba, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")

def plot_precision_recall_curve(y_true, y_proba, save_path):
    """Plot and save Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve saved to {save_path}")

def plot_feature_importance(model, feature_names, save_path, top_n=15):
    """Plot and save feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importance[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {save_path}")

def plot_threshold_analysis(y_true, y_proba, save_path):
    """Plot performance metrics across different thresholds."""
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred_threshold))
        precisions.append(precision_score(y_true, y_pred_threshold, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_threshold, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred_threshold, zero_division=0))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance Metrics vs Classification Threshold', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Threshold analysis plot saved to {save_path}")

def save_metrics_report(metrics, classification_rep, save_path):
    """Save comprehensive metrics report to text file."""
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DIABETES PREDICTION MODEL - EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 70 + "\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name:<40}: {value:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write("=" * 70 + "\n")
        f.write(classification_rep)
        f.write("\n" + "=" * 70 + "\n")
        
        f.write("\nMETRIC INTERPRETATIONS:\n")
        f.write("-" * 70 + "\n")
        f.write("• Accuracy: Overall correctness of predictions\n")
        f.write("• Precision: Of predicted diabetic cases, how many are correct\n")
        f.write("• Recall (Sensitivity): Of actual diabetic cases, how many are detected\n")
        f.write("• Specificity: Of actual non-diabetic cases, how many are detected\n")
        f.write("• F1-Score: Harmonic mean of precision and recall\n")
        f.write("• ROC-AUC: Area under ROC curve (0.5=random, 1.0=perfect)\n")
        f.write("• Average Precision: Area under Precision-Recall curve\n")
        f.write("• MCC: Correlation between predictions and truth (-1 to +1)\n")
        f.write("• Cohen Kappa: Agreement adjusted for chance\n")
        f.write("=" * 70 + "\n")
    
    print(f"Metrics report saved to {save_path}")

def main():
    """Main evaluation function."""
    print("\n" + "=" * 70)
    print("DIABETES PREDICTION MODEL EVALUATION")
    print("=" * 70 + "\n")
    
    # Load data and model
    X_test, X_test_scaled, y_test, model = load_data_and_model()
    
    # Make predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    classification_rep = classification_report(y_test, y_pred, 
                                               target_names=['No Diabetes', 'Diabetes'])
    
    # Display metrics
    print("\n" + "=" * 70)
    print("EVALUATION METRICS:")
    print("=" * 70)
    for metric_name, value in metrics.items():
        print(f"{metric_name:<40}: {value:.4f}")
    print("=" * 70)
    
    print("\nDetailed Classification Report:")
    print(classification_rep)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred, 
                         os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_proba, 
                  os.path.join(RESULTS_DIR, "roc_curve.png"))
    plot_precision_recall_curve(y_test, y_proba, 
                               os.path.join(RESULTS_DIR, "precision_recall_curve.png"))
    plot_feature_importance(model, EXPECTED_FEATURES, 
                           os.path.join(RESULTS_DIR, "feature_importance.png"))
    plot_threshold_analysis(y_test, y_proba, 
                           os.path.join(RESULTS_DIR, "threshold_analysis.png"))
    
    # Save comprehensive report
    save_metrics_report(metrics, classification_rep, 
                       os.path.join(RESULTS_DIR, "evaluation_report.txt"))
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print(f"All results saved to: {RESULTS_DIR}/")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
