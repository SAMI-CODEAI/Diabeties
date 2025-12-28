import os
import json
import re
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_data_from_pdf(pdf_path):
    """
    Main entry point for PDF extraction.
    """
    print(f"\n--- PDF Extraction Start: {pdf_path} ---")
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            extract = page.extract_text()
            if extract:
                text += extract + "\n"
        print(f"DEBUG: Successfully extracted {len(text)} characters of text.")
    except Exception as e:
        print(f"ERROR: Reading PDF file: {e}")
        return {}

    # Try LLM First
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            data = extract_with_openai(text, api_key)
            if data:
                print("DEBUG: Successfully extracted data using LLM.")
                return data
        except Exception as e:
            print(f"DEBUG: LLM Extraction failed: {e}. Falling back to Regex.")

    # Fallback to Regex
    data = extract_with_regex(text)
    print(f"DEBUG: Regex Extraction Results: {json.dumps(data, indent=2)}")
    return data

def extract_with_openai(text, api_key):
    """
    GPT-4o-mini parser with robust proxy-safe initialization.
    """
    import httpx
    # Use a clean client to avoid the "unexpected keyword argument 'proxies'" error
    client = OpenAI(api_key=api_key, http_client=httpx.Client())
    
    prompt = f"""
    Extract clinical metrics from this lab report.
    Fields: Glucose (mg/dL), BloodPressure (Diastolic ONLY), SkinThickness (mm), Insulin (mu U/ml), 
    BMI, Age, Gender, Pregnancies, DiabetesPedigreeFunction (float), Height (cm), Weight (kg).
    
    CRITICAL RULES:
    1. Do NOT extract Reference Ranges (e.g. 70-100). Extract the patient's ACTUAL RESULT.
    2. If Blood Pressure is 145/95, extract 95.
    3. If a value is missing, use null.
    4. Output ONLY valid JSON.
    
    Text:
    {text[:10000]} 
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    content = response.choices[0].message.content.strip()
    if "```json" in content:
        content = content.replace("```json", "").replace("```", "")
        
    return json.loads(content)

def extract_with_regex(text):
    """
    Hyper-robust regex extraction using multi-stage matching.
    """
    data = {}
    text_lower = text.lower()
    
    # Helper to clean numbers and avoid ranges
    def clean_val(m, group=1):
        if not m: return None
        val_str = m.group(group).strip()
        # If it's a range (contains '-' or 'to'), it's likely a reference range
        if '-' in val_str or ' to ' in val_str: return None
        try: return float(val_str)
        except: return None

    # 1. Glucose (Target Fasting)
    # Pattern 1: Label ... Result ... mg/dL
    # Pattern 2: Label ... mg/dL ... Result
    m = re.search(r'glucose.*?fasting.*?\b(\d{2,3}(?:\.\d+)?)\s*mg/dl', text_lower, re.DOTALL)
    if not m: m = re.search(r'glucose.*?fasting.*?mg/dl.*?\b(\d{2,3}(?:\.\d+)?)', text_lower, re.DOTALL)
    if not m: m = re.search(r'glucose.*?(\d{2,3}(?:\.\d+)?)', text_lower, re.DOTALL)
    data['Glucose'] = clean_val(m)

    # 2. Blood Pressure (Diastolic)
    m = re.search(r'diastolic.*?(\d{2,3})', text_lower, re.DOTALL)
    if not m:
        m = re.search(r'(?:bp|pressure).*?\d{2,3}\s*/\s*(\d{2,3})', text_lower, re.DOTALL)
    data['BloodPressure'] = clean_val(m)

    # 3. BMI
    m = re.search(r'bmi|body\s+mass\s+index.*?(\d{1,2}(?:\.\d+)?)', text_lower, re.DOTALL)
    data['BMI'] = clean_val(m)

    # 4. Age
    m = re.search(r'age.*?\b(\d{1,3})\b', text_lower)
    data['Age'] = clean_val(m)

    # 5. Skin Thickness
    m = re.search(r'skin\s+thickness.*?(\d{1,2}(?:\.\d+)?)', text_lower, re.DOTALL)
    if not m: m = re.search(r'skinfold.*?(\d{1,2}(?:\.\d+)?)', text_lower, re.DOTALL)
    data['SkinThickness'] = clean_val(m)

    # 6. Insulin
    m = re.search(r'insulin.*?(\d{1,3}(?:\.\d+)?)', text_lower, re.DOTALL)
    data['Insulin'] = clean_val(m)

    # 7. Diabetes Pedigree Function
    m = re.search(r'pedigree\s+function.*?\b(0\.\d{2,4})\b', text_lower, re.DOTALL)
    data['DiabetesPedigreeFunction'] = clean_val(m)

    # 8. Pregnancies
    m = re.search(r'pregnancies.*?(\d{1,2})\b', text_lower, re.DOTALL)
    data['Pregnancies'] = clean_val(m)

    # 9. Weight / Height
    m = re.search(r'weight.*?(\d{2,3}(?:\.\d+)?)', text_lower, re.DOTALL)
    data['Weight'] = clean_val(m)
    m = re.search(r'height.*?(\d{2,3}(?:\.\d+)?)', text_lower, re.DOTALL)
    data['Height'] = clean_val(m)

    # 10. Gender
    if 'female' in text_lower: data['Gender'] = 'Female'
    elif 'male' in text_lower: data['Gender'] = 'Male'

    return {k: v for k, v in data.items() if v is not None}
