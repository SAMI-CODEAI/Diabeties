"""Create a model usage heatmap
Rows: Papers + GlucoVision
Columns: LR, RF, SVM, DL, XAI
Values: 1 (used) or 0 (not used)
"""
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None

from literature_comparison.comparison.combined_dataframe import COMBINED_DF

# Helper to check presence of keywords

def used_contains(s, keywords):
    s = str(s).lower()
    return 1 if any(k.lower() in s for k in keywords) else 0

rows = []
for _, r in COMBINED_DF.iterrows():
    title = r.get('Paper Title', 'Unknown')
    algs = r.get('Algorithms / Models Used', '')
    dl = r.get('Deep Learning Used (Yes/No)', '')
    xai = r.get('Explainable AI Used (Yes/No)', '')
    row = {
        'Paper Title': title,
        'LR': used_contains(algs, ['Logistic Regression', 'Logistic']),
        'RF': used_contains(algs, ['Random Forest']),
        'SVM': used_contains(algs, ['SVM', 'Support Vector']),
        'DL': 1 if 'yes' in str(dl).lower() or 'lstm' in str(algs).lower() or 'cnn' in str(algs).lower() else 0,
        'XAI': 1 if 'yes' in str(xai).lower() or 'shap' in str(algs).lower() or 'lime' in str(algs).lower() else 0
    }
    rows.append(row)

mat_df = pd.DataFrame(rows).set_index('Paper Title')

# Improve figure sizing and tick labels
fig_h = max(6, 0.5 * len(mat_df))
plt.figure(figsize=(8, fig_h))
if sns is not None:
    ax = sns.heatmap(mat_df, annot=True, cbar=False, cmap='YlGnBu', linewidths=.5, linecolor='gray', fmt='d')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
else:
    plt.imshow(mat_df.values, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
    plt.yticks(range(len(mat_df.index)), mat_df.index, rotation=0)
    plt.xticks(range(len(mat_df.columns)), mat_df.columns, rotation=45, ha='right')
    # Annotate cells
    for i in range(mat_df.shape[0]):
        for j in range(mat_df.shape[1]):
            plt.text(j, i, str(mat_df.values[i, j]), ha='center', va='center', color='k')
plt.title('Model Usage Matrix (1=used, 0=not used)')
plt.tight_layout()
plt.savefig('literature_comparison/comparison/model_usage_heatmap.png', dpi=200)
plt.savefig('static/figures/model_usage_heatmap.png', dpi=200)
plt.close()

# Also expose the matrix as a DataFrame for downstream use
MODEL_USAGE_MATRIX = mat_df

if __name__ == '__main__':
    print(MODEL_USAGE_MATRIX)
    print("Heatmap saved at literature_comparison/comparison/model_usage_heatmap.png")
