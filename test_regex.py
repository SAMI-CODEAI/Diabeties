import re
import json

def extract_with_regex(text):
    data = {}
    text_lower = text.lower()
    
    def clean_val(m, group=1):
        if not m: return None
        val_str = m.group(group).strip()
        if '-' in val_str or ' to ' in val_str: return None
        try: return float(val_str)
        except: return None

    # 1. Age (Target "Age: 45" but avoid "Page: 1")
    m = re.search(r'\bage\s*(?:/\s*gender)?\s*[:\-]?\s*(\d{1,3})', text_lower)
    data['Age'] = clean_val(m)

    # 2. Glucose (Target Fasting result)
    # The value 138.0 appears after "mg/dL" in this report
    m = re.search(r'glucose.*?fasting.*?mg/dl.*?\b(\d{2,3}(?:\.\d+)?)\b', text_lower, re.DOTALL)
    if not m:
        m = re.search(r'glucose.*?fasting.*?\b(\d{2,3}(?:\.\d+)?)\s*mg/dl', text_lower, re.DOTALL)
    data['Glucose'] = clean_val(m)

    # 3. Blood Pressure (Diastolic)
    m = re.search(r'diastolic.*?(\d{2,3})', text_lower, re.DOTALL)
    if not m:
        m = re.search(r'(?:bp|pressure).*?\d{2,3}\s*/\s*(\d{2,3})', text_lower, re.DOTALL)
    data['BloodPressure'] = clean_val(m)

    # 4. BMI
    m = re.search(r'bmi|body\s+mass\s+index.*?(\d{1,2}(?:\.\d+)?)', text_lower, re.DOTALL)
    data['BMI'] = clean_val(m)

    # 5. Skin Thickness
    m = re.search(r'skin\s+thickness.*?(\d{1,3}(?:\.\d+)?)', text_lower, re.DOTALL)
    data['SkinThickness'] = clean_val(m)

    # 6. Insulin
    m = re.search(r'insulin.*?\b(\d{1,3}(?:\.\d+)?)\b', text_lower, re.DOTALL)
    data['Insulin'] = clean_val(m)

    # 7. Diabetes Pedigree Function
    m = re.search(r'diabetes\s+pedigree\s+function.*?(\d\.\d{2,4})', text_lower, re.DOTALL)
    data['DiabetesPedigreeFunction'] = clean_val(m)

    # 8. Pregnancies
    # Target "0 (Not Applicable)" or just a standalone digit near the label
    m = re.search(r'\bpregnancies\b.*?(\d{1,2})\s*\(not applicable\)', text_lower, re.DOTALL)
    if not m:
        # Fallback to looking for "0" or "None" if male
        if 'male' in text_lower:
            data['Pregnancies'] = 0.0
        else:
            m = re.search(r'\bpregnancies\b.*?\b(\d{1,2})\b', text_lower, re.DOTALL)
            data['Pregnancies'] = clean_val(m)
    else:
        data['Pregnancies'] = clean_val(m)

    # 9. Weight / Height
    m = re.search(r'height.*?\b(\d{2,3}(?:\.\d+)?)\s*cm', text_lower, re.DOTALL)
    data['Height'] = clean_val(m)
    m = re.search(r'weight.*?\b(\d{2,3}(?:\.\d+)?)\s*kg', text_lower, re.DOTALL)
    data['Weight'] = clean_val(m)

    # 10. Gender
    if 'female' in text_lower: data['Gender'] = 'Female'
    elif 'male' in text_lower: data['Gender'] = 'Male'

    return {k: v for k, v in data.items() if v is not None}

test_text = """
CITY CARE MULTISPECIALITY HOSPITAL
NABL Accredited Laboratory (MC-2841)
Endocrinology & Metabolic Research Division
Patient ID: CCH-992831
Report Type: Comprehensive Diabetes Profile
Page: 1 of 4
PATIENT CLINICAL SUMMARY
Patient Name: MR. RAJESH KUMAR
Age / Gender: 45 Years / Male
Ref. Doctor:
Pregnancies:
Dr. Anjali Sharma (MD)
Sample ID: 25122300441
Registered: 23-Dec-2025
UHID:
0 (Not Applicable)
1. PHYSICAL VITALS & BODY METRICS
Status:
Manual measurements recorded at the time of sample collection.
tableheader Metric Description
CCH-992831
Final Report
Value
Clinical Observation
Height
Weight
174 cm
95.0 kg
Body Mass Index (BMI)
Grade I Obesity
High Risk
31.4 kg/m2
Systolic Blood Pressure
145 mmHg
Diastolic Blood Pressure
Elevated
Stage 2 Hypertension
95 mmHg
2. PRIMARY GLYCEMIC STATUS
tableheader Test Compo
nent
Result
Units
Glucose Level (Fasting)
Glucose Level (Post Pran
dial)
mg/dL
138.0
Reference Range
70.0- 100.0
mg/dL
210.0
HbA1c
Hemoglobin)
(Glycated
¡ 140.0
%
4.0- 5.6
7.8
3. ENDOCRINE & INSULIN PROFILE
Advanced biochemical markers for insulin resistance analysis.
tableheader Parameter
Result
Units
Ref Range
Insulin (Fasting)
C-Peptide
µIU/mL
22.50
2.6- 24.9
3.10
ng/mL
1.1- 4.4
HOMA-IR (Calculated Index)
7.6
ratio
4. RESEARCH METRICS (METABOLIC DATA)
Additional parameters for diabetes pedigree and clinical research studies.
¡ 2.5
Skin Thickness (Triceps Skinfold):
The measured subcutaneous fat layer (Skin Thickness) is recorded at 32 mm.
Diabetes Pedigree Function (DPF):
Based on the patient’s reported multi-generational family history of Type 2 Diabetes (Maternal and
Paternal), the calculated **Diabetes Pedigree Function** is 0.651.
"""

results = extract_with_regex(test_text)
print(json.dumps(results, indent=2))
