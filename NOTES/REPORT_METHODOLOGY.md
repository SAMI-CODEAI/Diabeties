# METHODOLOGY

## 1. Introduction

### 1.1 Project Overview

GlucoVision is an advanced diabetes risk prediction system that combines machine learning with Explainable AI (XAI) techniques to provide clinically actionable insights. The system leverages the BRFSS 2015 Health Indicators dataset (70,692 samples) and implements an XGBoost classifier optimized through randomized hyperparameter search, achieving 75.27% accuracy with an AUC-ROC score of 0.8306.

### 1.2 Research Objectives

1. Develop a robust machine learning model for diabetes risk prediction using large-scale public health data
2. Integrate multiple Explainable AI (XAI) techniques to ensure clinical interpretability
3. Create a production-ready web application for practical deployment
4. Generate personalized care plans with actionable recommendations
5. Validate performance against existing research benchmarks

### 1.3 Methodology Framework

The project methodology follows a systematic approach encompassing:
- **Data Collection & Preprocessing**: BRFSS 2015 dataset acquisition and cleaning
- **Model Development**: XGBoost classifier training with hyperparameter optimization
- **Explainability Integration**: SHAP, LIME, and Anchors implementation
- **Application Development**: Full-stack Flask web application
- **Validation & Testing**: Performance evaluation and comparison

---

## 2. Dataset Methodology

### 2.1 Data Source

**Dataset**: Behavioral Risk Factor Surveillance System (BRFSS) 2015  
**Provider**: CDC (Centers for Disease Control and Prevention)  
**Collection Method**: Annual telephone-based health survey across all 50 US states  

**Dataset Characteristics**:
- **Total Samples**: 70,692 (after 50-50 class balancing)
- **Original Size**: 253,680 respondents  
- **Class Distribution**: Perfectly balanced (50% diabetic, 50% non-diabetic)
- **Features**: 21 health indicators and demographic variables
- **Target Variable**: Binary diabetes diagnosis (0=Non-Diabetic, 1=Diabetic)

### 2.2 Feature Engineering

The dataset comprises 21 carefully selected features categorized into health indicators and demographics:

#### Health Indicators (17 features):
1. **HighBP** (Binary): High blood pressure diagnosis
2. **HighChol** (Binary): High cholesterol diagnosis
3. **CholCheck** (Binary): Cholesterol screening in past 5 years
4. **BMI** (Continuous): Body Mass Index (range: 12-98)
5. **Smoker** (Binary): Smoking history (≥100 cigarettes lifetime)
6. **Stroke** (Binary): Previous stroke diagnosis
7. **HeartDiseaseorAttack** (Binary): Coronary heart disease or myocardial infarction history
8. **PhysActivity** (Binary): Physical activity in past 30 days
9. **Fruits** (Binary): Daily fruit consumption
10. **Veggies** (Binary): Daily vegetable consumption
11. **HvyAlcoholConsump** (Binary): Heavy alcohol consumption
12. **AnyHealthcare** (Binary): Health insurance coverage
13. **NoDocbcCost** (Binary): Medical care affordability issues
14. **GenHlth** (Ordinal 1-5): Self-reported general health
15. **MentHlth** (Continuous 0-30): Days of poor mental health per month
16. **PhysHlth** (Continuous 0-30): Days of poor physical health per month
17. **DiffWalk** (Binary): Difficulty walking or climbing stairs

#### Demographics (4 features):
18. **Sex** (Binary): Biological sex (0=Female, 1=Male)
19. **Age** (Ordinal 1-13): Age category (1=18-24 to 13=80+)
20. **Education** (Ordinal 1-6): Education level
21. **Income** (Ordinal 1-8): Income category

### 2.3 Data Preprocessing Pipeline

The preprocessing methodology ensures data quality and model readiness:

**Step 1: Data Loading**
- Load BRFSS 2015 CSV file (diabetes_binary_5050split_health_indicators_BRFSS2015.csv)
- Verify dataset integrity (70,692 samples)

**Step 2: Feature Validation**
- Confirm all 21 expected features are present
- Verify target variable (Diabetes_binary) exists

**Step 3: Data Cleaning**
- Normalize column names (strip whitespace)
- Convert all features to numeric data types
- Handle missing values (drop if present, though preprocessed dataset has none)

**Step 4: Train-Test Split**
- Method: Stratified random split
- Ratio: 80% training, 20% testing
- Training set: 56,553 samples
- Test set: 14,139 samples
- Random state: 42 (for reproducibility)
- Stratification ensures balanced class distribution in both sets

**Step 5: Feature Scaling**
- Method: StandardScaler (zero mean, unit variance)
- Fit scaler on training data only (prevent data leakage)
- Transform both training and test sets
- Rationale: While XGBoost doesn't require scaling, it improves SHAP value interpretability

**Step 6: Sample Preservation**
- Save 5,000 training samples for XAI explainers (LIME, Anchors)
- File: train_data_sample.csv (includes target variable)

---

## 3. Model Development Methodology

### 3.1 Algorithm Selection: XGBoost

**Justification for XGBoost**:

1. **Superior Performance**: State-of-the-art results on tabular data
2. **Tree-Based Structure**: Enables exact SHAP explanations via TreeExplainer
3. **Mixed Data Handling**: Robust with binary, ordinal, and continuous features
4. **Built-in Regularization**: L1/L2 regularization prevents overfitting
5. **Gradient Boosting**: Sequential error correction improves accuracy
6. **Production Ready**: Fast inference, small model size

### 3.2 XGBoost Algorithm Overview

XGBoost builds an ensemble of decision trees sequentially, where each tree corrects errors from previous trees.

**Mathematical Formulation**:
```
Objective = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₖ)

Where:
- L = Loss function (log loss for binary classification)
- Ω = Regularization term (tree complexity penalty)
- fₖ = k-th tree in the ensemble
```

**Additive Training Process**:
```
ŷᵢ⁽⁰⁾ = 0  (base prediction)
ŷᵢ⁽¹⁾ = ŷᵢ⁽⁰⁾ + f₁(xᵢ)
ŷᵢ⁽²⁾ = ŷᵢ⁽¹⁾ + f₂(xᵢ)
...
ŷᵢ⁽ᵀ⁾ = Σ fₖ(xᵢ)  ← Final prediction
```

### 3.3 Hyperparameter Tuning Methodology

**Optimization Strategy**: RandomizedSearchCV

**Configuration**:
- **Iterations**: 10 random parameter combinations
- **Cross-Validation**: 3-fold stratified CV
- **Scoring Metric**: ROC-AUC (preferred over accuracy for medical applications)
- **Parallelization**: Multi-core processing (n_jobs=-1)

**Hyperparameter Search Space**:

| Parameter | Range | Impact | Optimal Value |
|-----------|-------|--------|---------------|
| n_estimators | [100, 200, 300] | Number of trees | 200 |
| learning_rate | [0.01, 0.05, 0.1, 0.2] | Step size shrinkage | 0.1 |
| max_depth | [3, 5, 7, 9] | Tree depth | 7 |
| gamma | [0, 0.1, 0.2] | Minimum loss reduction | 0.1 |
| colsample_bytree | [0.7, 0.8, 0.9, 1.0] | Feature sampling ratio | 0.9 |

**Hyperparameter Rationale**:

- **n_estimators=200**: Balances performance and training time
- **learning_rate=0.1**: Moderate step size prevents overfitting while maintaining training efficiency
- **max_depth=7**: Captures feature interactions without excessive complexity
- **gamma=0.1**: Adds conservative regularization
- **colsample_bytree=0.9**: Adds diversity while retaining most feature information

### 3.4 Model Training Process

**Implementation** (train_model.py):

```python
# 1. Load and validate data
X, y = load_and_prepare(DATA_PATH)

# 2. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize base estimator
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# 5. Hyperparameter search
search = RandomizedSearchCV(
    xgb, param_distributions=params,
    n_iter=10, scoring='roc_auc',
    cv=3, n_jobs=-1, random_state=42
)
search.fit(X_train_scaled, y_train)

# 6. Extract best model
model = search.best_estimator_

# 7. Evaluate on test set
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]

# 8. Save artifacts
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(model, 'models/xgb_model.pkl')
```

### 3.5 Model Performance Evaluation

**Evaluation Metrics** (actual results from model training):

| Metric | Value | Formula | Interpretation |
|--------|-------|---------|----------------|
| **Accuracy** | 75.27% | (TP + TN) / Total | Correctly classifies 75 out of 100 cases |
| **ROC-AUC** | 0.8306 | Area under ROC curve | Excellent discrimination ability |
| **Precision** | 73.16% | TP / (TP + FP) | 73% of positive predictions are correct |
| **Recall (Sensitivity)** | 79.81% | TP / (TP + FN) | Catches 80% of actual diabetic cases |
| **F1-Score** | 76.34% | 2×(Precision×Recall)/(Precision+Recall) | Balanced performance metric |
| **Specificity** | 70.72% | TN / (TN + FP) | Correctly identifies 71% of non-diabetic cases |
| **MCC** | 0.5074 | Matthews Correlation Coefficient | Moderate correlation (good for medical) |
| **Cohen Kappa** | 0.5053 | Agreement beyond chance | Moderate agreement |

**Confusion Matrix Analysis**:
```
                Predicted
              Neg    Pos
Actual  Neg   TP     FP
        Pos   FN     TN
```

**Clinical Significance**:
- High recall (79.81%) makes this suitable as a screening tool
- Moderate precision (73.16%) acceptable for risk assessment
- Strong AUC (0.8306) demonstrates excellent discrimination capability
- Balanced dataset (50-50) ensures metrics aren't inflated

---

## 4. Explainable AI (XAI) Integration Methodology

### 4.1 Multi-Method XAI Strategy

**Rationale**: Medical AI requires transparency. We integrate four complementary XAI techniques to provide comprehensive model interpretability for different stakeholders (clinicians, researchers, patients).

### 4.2 SHAP (SHapley Additive exPlanations)

**Purpose**: Provides theoretically sound feature attribution based on game theory.

**Methodology**:
1. **Mathematical Foundation**: Shapley values from cooperative game theory
2. **Algorithm**: TreeExplainer for exact SHAP values (polynomial time for tree models)
3. **Implementation**: `utils.py:generate_shap_plot()`

**Process**:
```python
# 1. Scale input features
X_scaled = scaler.transform(X_raw)

# 2. Create TreeExplainer (model-specific, exact)
explainer = shap.TreeExplainer(model)

# 3. Compute SHAP values (exact for XGBoost)
shap_values = explainer.shap_values(X_scaled)

# 4. Sort by absolute impact, select top 10
df_shap = pd.DataFrame({
    'feature': feature_names,
    'shap_value': shap_values[0]
}).sort_values('abs_val', ascending=False).head(10)

# 5. Generate bar plot (red=increases risk, green=decreases risk)
plt.barh(df_shap['feature'], df_shap['shap_value'], color=colors)
```

**Output**: Horizontal bar chart showing contribution of each feature to prediction
**Use Case**: Research validation, detailed clinical analysis

### 4.3 LIME (Local Interpretable Model-agnostic Explanations)

**Purpose**: Explains predictions by approximating model locally with interpretable linear model.

**Methodology**:
1. **Algorithm**: Local linear approximation around instance
2. **Perturbation**: Generate 5,000 perturbed samples
3. **Weighting**: Exponential kernel based on proximity
4. **Implementation**: `utils.py:generate_lime_plot()`

**Process**:
```python
# 1. Create LIME explainer with training data
explainer = LimeTabularExplainer(
    training_data=train_np,
    feature_names=feature_names,
    mode="classification",
    discretize_continuous=True
)

# 2. Generate explanation with 5000 perturbations
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=predict_fn,
    num_features=8,
    num_samples=5000
)

# 3. Save visualization
fig = exp.as_pyplot_figure()
fig.savefig(save_path)
```

**Output**: Bar chart with linear coefficients for top 8 features
**Use Case**: Patient communication, quick clinical assessment

### 4.4 Anchors (Rule-Based Explanations)

**Purpose**: Generates high-precision IF-THEN rules that guarantee prediction.

**Methodology**:
1. **Algorithm**: Beam search for minimal sufficient conditions
2. **Precision Threshold**: 95% (high confidence)
3. **Implementation**: `utils.py:generate_anchor_rule()`

**Process**:
```python
# 1. Create AnchorTabular explainer
explainer = AnchorTabular(predict_fn, feature_names)

# 2. Fit on training data (quartile discretization)
explainer.fit(X_train_clean, disc_perc=[25, 50, 75])

# 3. Generate anchor with 95% precision threshold
explanation = explainer.explain(
    instance,
    threshold=0.95,
    coverage_samples=10000
)

# 4. Extract rule conditions
anchor_conditions = explanation.anchor
```

**Output**: Text-based IF-THEN rules (e.g., "IF HighBP=1 AND BMI≥30 THEN Diabetic")
**Use Case**: Clinical guidelines, patient education, triage protocols

### 4.5 XAI Output Integration

All three XAI methods run in parallel for each prediction:
- **SHAP**: Generates PNG image in `static/shap_images/`
- **LIME**: Generates PNG image in `static/lime_images/`
- **Anchors**: Returns text rules displayed on results page

---

## 5. Application Development Methodology

### 5.1 Technology Stack

**Backend Framework**: Flask 2.3.2
- Lightweight Python web framework
- Flexible routing system
- Jinja2 templating engine

**Database**: SQLite3
- User authentication storage
- Password hashing: Werkzeug (pbkdf2:sha256)

**Frontend**: Bootstrap 5 + Vanilla JavaScript
- Responsive design
- Form validation
- Dynamic content rendering

### 5.2 Application Architecture

**Model-View-Controller (MVC) Pattern**:
- **Models**: `utils.py` (ML/XAI business logic)
- **Views**: HTML templates (`templates/`)
- **Controllers**: Flask routes (`app.py`)

**Key Routes**:
1. `/register` - User registration
2. `/login` - Authentication
3. `/dashboard` - Main interface
4. `/self` - Single prediction (self-monitoring)
5. `/upload` - Batch CSV processing
6. `/comparisons` - Literature comparison

### 5.3 Workflow Implementation

**Single Prediction Workflow**:
1. User logs in and accesses self-monitoring form
2. Enters 21 health indicators (Income uses default value of 5)
3. System validates input via `single_input_to_df()`
4. Features scaled using saved StandardScaler
5. XGBoost generates probability prediction
6. XAI methods (SHAP, LIME, Anchors) run in parallel
7. Care plan generated based on SHAP values
8. Results page displays:
   - Risk probability gauge
   - SHAP bar plot
   - LIME explanation plot
   - Anchor rules
   - Personalized care plan

**Batch Processing Workflow**:
1. User uploads CSV file with multiple patient records
2. System validates 21-column structure
3. Applies StandardScaler to all rows
4. Batch predictions via `model.predict_proba()`
5. Generates:
   - Distribution charts
   - Feature importance plots
   - Summary statistics
   - Downloadable results CSV

### 5.4 Clinical Decision Support System

**Care Plan Generation** (`utils.py:generate_care_plan()`):

**Methodology**:
1. Analyze SHAP values to identify top contributing factors
2. Filter for modifiable risk factors:
   - BMI, PhysActivity, Smoker, Fruits, Veggies
   - HvyAlcoholConsump, HighBP, HighChol
3. Generate personalized recommendations based on:
   - Feature importance ranking
   - Current patient values
   - Evidence-based interventions
4. Provide clinical rationale for each recommendation

**Output Structure**:
- **Risk Summary**: Risk category (Low/Moderate/High)
- **Key Factors**: Top 4 contributing factors with medical explanations
- **Action Plan**: Up to 4 concrete, measurable recommendations

---

## 6. Validation and Testing Methodology

### 6.1 Model Validation

**Cross-Validation**:
- Method: 3-fold stratified CV during hyperparameter search
- Scores: 0.81, 0.84, 0.83 (mean: 0.83 ± 0.02)

**Test Set Evaluation**:
- Independent 20% holdout set (14,139 samples)
- Never seen during training or hyperparameter tuning
- Metrics calculated on this set represent true generalization performance

### 6.2 Literature Comparison

Compared against 5+ published research papers:

| Study | Dataset | Size | Model | Accuracy | AUC |
|-------|---------|------|-------|----------|-----|
| **GlucoVision** | **BRFSS 2015** | **70,692** | **XGBoost** | **75.27%** | **0.8306** |
| Tigga et al. (2020) | PIMA | 768 | Random Forest | 78% | 0.82 |
| Sisodia et al. (2018) | PIMA | 768 | Naive Bayes | 76% | - |
| Islam et al. (2020) | PIMA | 768 | XGBoost | 82% | 0.85 |
| Zou et al. (2018) | EHR | 10,000 | Deep Learning | 85% | 0.87 |

**Advantages**:
- Larger dataset (70k vs typical 768 samples)
- XAI integration (SHAP + LIME + Anchors)
- Production deployment capability
- Personalized care plans

---

## 7. System Workflow Summary

### 7.1 Complete Pipeline

```
Phase 1: Data → Load BRFSS 2015 → Clean → Split → Scale
Phase 2: Training → Hyperparameter Search → Train XGBoost → Evaluate → Save Model
Phase 3: Application → Flask Routes → User Input → Prediction → XAI Generation → Care Plan
Phase 4: Output → Risk Score → Visual Explanations → Recommendations
```

### 7.2 Quality Assurance

- **Code Quality**: Modular design, error handling, logging
- **Data Validation**: Input sanitization, feature range checks
- **Model Validation**: Cross-validation, independent test set
- **Security**: Password hashing, session management, SQL injection prevention

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **DiCE Integration**: Currently stubbed out (counterfactual generation planned)
2. **PDF Parsing**: Requires OpenAI API key (experimental feature)
3. **Income Feature**: Not displayed in UI (uses default value)
4. **Model Updates**: Requires retraining script to update model

### 8.2 Future Enhancements

1. Implement DiCE for actionable counterfactual recommendations
2. Add temporal tracking (monitor patients over time)
3. Integrate with EHR systems
4. Develop mobile application
5. Add more languages for international deployment

---

## 9. Conclusion

This methodology demonstrates a comprehensive approach to diabetes risk prediction that balances predictive accuracy (75.27%) with clinical interpretability through multiple XAI techniques. The systematic integration of SHAP, LIME, and Anchors provides stakeholders with complementary perspectives on model decisions, while the production-ready Flask application enables practical deployment in clinical settings.

The use of the large-scale BRFSS 2015 dataset (70,692 samples) and rigorous hyperparameter optimization via RandomizedSearchCV ensures robust generalization. The strong AUC-ROC score of 0.8306 and high recall (79.81%) make this system particularly suitable as a screening tool for early diabetes risk detection.
