"""Combine all paper dataframes and the GlucoVision project dataframe into a unified DataFrame
Saves combined CSV at `literature_comparison/comparison/combined_comparison.csv` and exposes `COMBINED_DF`.
"""
import pandas as pd

# Import paper dataframes
from literature_comparison.papers.paper_01_ieee_2019 import PAPER_DF as P1
from literature_comparison.papers.paper_02_elsevier_2020 import PAPER_DF as P2
from literature_comparison.papers.paper_03_nature_2021 import PAPER_DF as P3
from literature_comparison.papers.paper_04_ieee_xplore_2020 import PAPER_DF as P4
from literature_comparison.papers.paper_05_springer_2018 import PAPER_DF as P5
from literature_comparison.papers.paper_06_procedia_2017 import PAPER_DF as P6
from literature_comparison.papers.paper_07_springer_dl_2020 import PAPER_DF as P7
from literature_comparison.papers.paper_08_iot_ieee_2021 import PAPER_DF as P8
from literature_comparison.papers.paper_09_xai_shap_2022 import PAPER_DF as P9
from literature_comparison.papers.paper_10_acm_survey_2021 import PAPER_DF as P10

# Import project dataframe
from literature_comparison.my_project.glucovision_project import PROJECT_DF

ALL_DFS = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, PROJECT_DF]
COMBINED_DF = pd.concat(ALL_DFS, ignore_index=True)

# Ensure column ordering and add convenience fields
if "Accuracy" not in COMBINED_DF.columns:
    # Try to parse numeric accuracy from 'Accuracy / Performance' if present (best-effort)
    COMBINED_DF["Accuracy"] = None

# Save CSV and pickle (pandas DataFrame object saved via pickle)
OUT_CSV = "literature_comparison/comparison/combined_comparison.csv"
OUT_PKL = "literature_comparison/comparison/combined_comparison.pkl"
COMBINED_DF.to_csv(OUT_CSV, index=False)
COMBINED_DF.to_pickle(OUT_PKL)

# When imported, COMBINED_DF will be available for plots and further analysis.

if __name__ == "__main__":
    print("Combined DataFrame saved to:", OUT_CSV)
    print(COMBINED_DF.head())
