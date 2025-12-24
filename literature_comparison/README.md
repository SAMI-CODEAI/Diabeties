# Literature Comparison — GlucoVision (Master's-level literature synthesis)

## 1. What makes GlucoVision unique ✅

GlucoVision unifies classical machine learning and contemporary deep learning architectures (LR, RF, SVM, LSTM, GRU, CNN) within a single reproducible pipeline and pairs these models with per-instance explainability (SHAP and LIME). The project additionally emphasizes deployment-readiness via a Streamlit web application, enabling clinician-facing interpretability and real-time usage.

## 2. How GlucoVision improves upon earlier works 🔧
- Integrates both classical and deep learning approaches and reports comparative performance in the **same** experimental framework, allowing fair cross-method evaluation.
- Provides **per-instance** explanations (SHAP + LIME) rather than only global feature importance, addressing a major gap in clinical interpretability.
- Delivers a deployment artifact (Streamlit app) and reproducible scripts for the full pipeline (training → explanation → deployment), thereby lowering the barrier for translational evaluation.

## 3. Research gaps addressed 💡
- Bridging accuracy and interpretability: employing XAI methods to make high-performing models (including DL) actionable in clinical settings.
- Reproducibility and deployment: shipping a runnable prototype and providing dataset+model artifacts for validation.
- Multi-model validation: direct comparisons across LR, RF, SVM and DL (LSTM/GRU/CNN) in one project with consistent preprocessing and evaluation.

## 4. Why XAI + Deployment is critical ⚠️
Explainability fosters trust and enables case-level validation by clinicians, which is essential in high-stakes domains such as diabetes risk prediction. Deployment as a web app streamlines prospective validation, user testing, and integration into clinical workflows.

## 5. Summary (suitable for IEEE Chapter 2: Literature Review) 📚
The corpus of diabetes prediction literature demonstrates an evolution from classical machine learning methods (2016–2020), through the adoption of deep learning on longitudinal EHR data (2020), to an increased emphasis on explainable AI (2021–2022) for clinical deployment. GlucoVision situates itself at the intersection of these developments by providing a rigorous comparative evaluation, integrating per-instance XAI, and delivering a deployment-ready application—addressing both methodological and translational gaps identified in prior work.

---

For reproducibility: the `comparison/` folder contains scripts to combine the paper metadata frames, generate comparison plots, and create a model usage heatmap.
