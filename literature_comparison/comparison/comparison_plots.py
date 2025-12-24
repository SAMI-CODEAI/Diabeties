"""Generate comparison plots for the literature comparison:
1. Bar chart comparing accuracy
2. Bar chart comparing XAI usage (counts)
3. Timeline graph showing evolution: Traditional ML -> Deep Learning -> Explainable AI

Plots are saved into `literature_comparison/comparison/`.
"""
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set(style="whitegrid")
except Exception:
    sns = None
import pandas as pd

from literature_comparison.comparison.combined_dataframe import COMBINED_DF

# Prepare a working df copy
df = COMBINED_DF.copy()

# 1) Bar chart comparing Accuracy (use 'Accuracy' numeric column if available)
if "Accuracy" in df.columns and pd.api.types.is_numeric_dtype(df["Accuracy"]):
    acc_df = df[["Paper Title", "Accuracy"]].dropna().sort_values("Accuracy", ascending=False)
else:
    # Try to parse best-effort numeric accuracy from the 'Accuracy / Performance' string
    def extract_numeric(x):
        try:
            import re
            m = re.search(r"(0\.[5-9]\d)|(0\.(?:7|8|9)\d)", str(x))
            if m:
                return float(m.group(0))
        except Exception:
            pass
        return None
    df["_acc_parsed"] = df["Accuracy / Performance"].apply(extract_numeric)
    acc_df = df[["Paper Title", "_acc_parsed"]].dropna().rename(columns={"_acc_parsed": "Accuracy"}).sort_values("Accuracy", ascending=False)

# Horizontal bar chart with wrapped labels and dynamic height
import textwrap
n = acc_df.shape[0]
fig_h = max(6, n * 0.6)
plt.figure(figsize=(10, fig_h))
# Wrap titles longer than 40 chars
acc_df['_title_wrapped'] = acc_df['Paper Title'].apply(lambda t: textwrap.fill(t, width=40))
if sns is not None:
    sns.barplot(data=acc_df, x="Accuracy", y="_title_wrapped", palette="Blues_d")
else:
    plt.barh(acc_df['_title_wrapped'], acc_df["Accuracy"], color='steelblue')
    plt.gca().invert_yaxis()
plt.xlabel('Accuracy (numeric)')
plt.title("Accuracy / Performance Comparison Across Papers")
plt.xlim(0,1)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
# Save both local comparison and static folder for web app
plt.savefig("literature_comparison/comparison/accuracy_comparison.png", dpi=200)
plt.savefig("static/figures/accuracy_comparison.png", dpi=200)
plt.close()

# 2) Bar chart comparing Explainable AI usage (count of Yes/No)
# Clean Explainable AI Used column for plotting
def clean_xai(val):
    s = str(val).lower()
    if "yes" in s:
        return "Yes"
    elif "no" in s:
        return "No"
    return "Unknown"

xai_counts = df["Explainable AI Used (Yes/No)"].apply(clean_xai).value_counts()
plt.figure(figsize=(6,4))
if sns is not None:
    sns.barplot(x=xai_counts.index, y=xai_counts.values, palette="muted")
else:
    plt.bar(xai_counts.index, xai_counts.values, color=['#4c72b0','#c44e52'])
plt.title("Explainable AI Usage (Yes vs No)")
plt.ylabel("Number of Papers")
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("literature_comparison/comparison/xai_usage_counts.png", dpi=200)
plt.savefig("static/figures/xai_usage_counts.png", dpi=200)
plt.close()

# 2.5) Detailed XAI per paper (Qualitative Text Plot)
# Extract the technique text
def extract_technique(val):
    val = str(val).strip()
    if val.lower() == "no":
        return "None"
    # Remove 'Yes' or 'No' and parens to get the meat
    # e.g. "Yes (SHAP)" -> "SHAP"
    # e.g. "Partial (feature importance)" -> "Feature Importance"
    import re
    m = re.search(r"\((.*?)\)", val)
    if m:
        return m.group(1)
    if "Yes" in val:
        return "Unspecified XAI"
    return val

df["XAI_Technique"] = df["Explainable AI Used (Yes/No)"].apply(extract_technique)
# Plot as a table-like horizontal bar chart
n_papers = len(df)
fig_h_xai = max(6, n_papers * 0.6)
plt.figure(figsize=(10, fig_h_xai))

# Use wrapped titles
df['_title_wrapped'] = df['Paper Title'].apply(lambda t: textwrap.fill(t, width=40))

# We'll plot a simple scatter or barh with text
# Create a color mapping based on 'clean_xai' result
df['XAI_Status'] = df["Explainable AI Used (Yes/No)"].apply(clean_xai)
palette = {"Yes": "#2ca02c", "No": "#d62728", "Unknown": "#7f7f7f"}
colors = df['XAI_Status'].map(palette).fillna("#7f7f7f")

# Vertical positions
y_pos = range(n_papers)
# Sort by XAI status so Yes are together? Or keep original order? 
# Let's sort by XAI Status then Title to group them
df_sorted = df.sort_values(["XAI_Status", "Paper Title"], ascending=[False, True])

plt.barh(df_sorted['_title_wrapped'], [1]*n_papers, color=df_sorted['XAI_Status'].map(palette).fillna("#7f7f7f"), alpha=0.2)
# Add text labels for the techniques
for i, (idx, row) in enumerate(df_sorted.iterrows()):
    tech = row['XAI_Technique']
    plt.text(0.05, i, tech, va='center', fontsize=10, fontweight='bold', color='black')

plt.xlim(0, 1.2)
plt.xticks([]) # Hide x axis
plt.title("XAI Techniques Used per Paper")
plt.xlabel("Technique")
plt.tight_layout()
plt.savefig("literature_comparison/comparison/xai_per_paper.png", dpi=200)
plt.savefig("static/figures/xai_per_paper.png", dpi=200)
plt.close()

# 3) Timeline graph showing evolution: Traditional ML -> Deep Learning -> Explainable AI
# Create indicator columns
def bool_to_int_yes(x):
    if isinstance(x, str):
        return 1 if "Yes" in x or "yes" in x or "yes" in x.lower() else 0
    return 1 if bool(x) else 0

# Traditional ML if Algorithms contain LR, RF, SVM
import numpy as np

df["HasTraditionalML"] = df["Algorithms / Models Used"].apply(lambda s: 1 if any(k in str(s) for k in ["Logistic", "Random Forest", "SVM", "Decision Tree", "KNN"]) else 0)
df["HasDL"] = df["Deep Learning Used (Yes/No)"].apply(lambda s: 1 if "Yes" in str(s) else 0)
df["HasXAI"] = df["Explainable AI Used (Yes/No)"].apply(lambda s: 1 if "Yes" in str(s) else 0)

# Melt to long form for timeline plotting
timeline = df[["Paper Title", "Publication Year", "HasTraditionalML", "HasDL", "HasXAI"]].copy()

# EXCLUDE THIS PROJECT (GlucoVision)
timeline = timeline[~timeline["Paper Title"].str.contains("GlucoVision", case=False, na=False)]

# Normalize Publication Year to int where possible (best-effort)
import re

def parse_year(y):
    try:
        return int(re.search(r"(19|20)\d{2}", str(y)).group(0))
    except Exception:
        return np.nan

timeline["YearInt"] = timeline["Publication Year"].apply(parse_year)

timeline_melt = timeline.melt(id_vars=["Paper Title", "YearInt"], value_vars=["HasTraditionalML","HasDL","HasXAI"], var_name="Category", value_name="Used")
# Only keep Used==1
timeline_melt = timeline_melt[timeline_melt["Used"]==1]

# Map category display names
cat_names = {"HasTraditionalML": "Traditional ML", "HasDL": "Deep Learning", "HasXAI": "Explainable AI"}
timeline_melt["Technique Type"] = timeline_melt["Category"].map(cat_names)

# Plotting with Seaborn for better look
plt.figure(figsize=(12, 6))

if sns is not None:
    # Use stripplot for jitter
    sns.stripplot(
        data=timeline_melt, 
        x="YearInt", 
        y="Technique Type", 
        hue="Technique Type",
        order=["Traditional ML", "Deep Learning", "Explainable AI"],
        palette={"Traditional ML": "#FBB4AE", "Deep Learning": "#B3CDE3", "Explainable AI": "#CCEBC5"},
        size=10,
        linewidth=1,
        alpha=0.8,
        jitter=0.2
    )
else:
    # Fallback to scatter
    cat_map = {"Traditional ML": 1, "Deep Learning": 2, "Explainable AI": 3}
    for cat, group in timeline_melt.groupby('Technique Type'):
        plt.scatter(group["YearInt"], [cat_map[cat]]*len(group), label=cat, s=100, alpha=0.7)
    plt.yticks([1,2,3], ["Traditional ML", "Deep Learning", "Explainable AI"])

plt.xlabel("Publication Year", fontsize=12, fontweight='bold')
plt.ylabel("")
plt.title("Evolution of Diabetes Prediction Techniques", fontsize=14, fontweight='bold', pad=20)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Add a subtle background arrow or annotation to show flow
start_year = timeline_melt['YearInt'].min()
end_year = timeline_melt['YearInt'].max()
plt.xlim(start_year - 1, end_year + 1)

# Annotate trends
plt.text(start_year, 2.5, "Rising trend of XAI", fontsize=10, style='italic', color='grey')
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig("literature_comparison/comparison/timeline_evolution.png", dpi=200)
plt.savefig("static/figures/timeline_evolution.png", dpi=200)
plt.close()

print("Plots saved to literature_comparison/comparison/")
