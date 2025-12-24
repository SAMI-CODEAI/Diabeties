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
    Extracts clinical metrics from a PDF report using OpenAI (RAG approach)
    with a fallback to Regex heuristics if API key is missing or fails.
    """
    print(f"Extracting data from: {pdf_path}")
    
    # 1. Extract Text from PDF
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            extract = page.extract_text()
            if extract:
                text += extract + "\n"
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return {}

    # 2. Try LLM Extraction
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            return extract_with_openai(text, api_key)
        except Exception as e:
            print(f"LLM Extraction failed: {e}. Falling back to regex.")
    else:
        print("OPENAI_API_KEY not found. Using regex fallback.")

    # 3. Fallback to Regex
    return extract_with_regex(text)

def extract_with_openai(text, api_key):
    """
    Uses OpenAI GPT-4o-mini (or similar) to parse the text and extract structured data.
    """
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    You are an expert medical data extractor. Your task is to extract specific clinical metrics from the following lab report text.
    
    Extract the following fields:
    - Glucose (mg/dL)
    - BloodPressure (mm Hg) - If given as Systolic/Diastolic (e.g., 120/80), return the Diastolic value (80).
    - SkinThickness (mm) - Triceps skin fold thickness.
    - Insulin (mu U/ml)
    - BMI (Body Mass Index)
    - Age (years)
    - Gender (Male/Female)
    - Pregnancies (count)
    - DiabetesPedigreeFunction (float)
    - Height (cm) - Optional
    - Weight (kg) - Optional
    
    Format the output as a valid JSON object with keys: "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age", "Gender", "Pregnancies", "DiabetesPedigreeFunction", "Height", "Weight".
    
    Rules:
    - If a value is not found, set it to null.
    - Return ONLY the JSON object, no markdown formatting.
    
    Text content:
    {text[:10000]} 
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini", # cheap and fast
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant that outputs JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    content = response.choices[0].message.content.strip()
    
    # Clean up response to get JSON if md ticks exist
    if "```json" in content:
        content = content.replace("```json", "").replace("```", "")
        
    try:
        data = json.loads(content)
        # Filter out nulls
        return {k: v for k, v in data.items() if v is not None}
    except json.JSONDecodeError:
        print("Failed to decode JSON from OpenAI response")
        return {}

def extract_with_regex(text):
    """
    Legacy regex extraction.
    """
    data = {}
    text_lower = text.lower()
    
    # 1. Glucose
    glucose_match = re.search(r'(?:glucose|sugar|fbs|rbs|fasting).{0,20}?[:\-\s]\s*(\d{2,3}(?:\.\d)?)', text_lower)
    if glucose_match:
        data['Glucose'] = float(glucose_match.group(1))
    
    # 2. Age
    age_match = re.search(r'age\s*[:\-\s]\s*(\d{1,3})', text_lower)
    if age_match:
        data['Age'] = float(age_match.group(1))

    # 3. Blood Pressure (Diastolic)
    bp_match = re.search(r'pressure|bp\s*.{0,10}[:\-\s]\s*(\d{2,3})\s*/\s*(\d{2,3})', text_lower)
    if bp_match:
        data['BloodPressure'] = float(bp_match.group(2)) # Diastolic
    
    # 4. Weight
    weight_match = re.search(r'weight\s*[:\-\s]\s*(\d{2,3}(?:\.\d)?)', text_lower)
    if weight_match:
        data['Weight'] = float(weight_match.group(1))

    # 5. BMI
    bmi_match = re.search(r'bmi|body mass index\s*[:\-\s]\s*(\d{1,2}(?:\.\d)?)', text_lower)
    if bmi_match:
        data['BMI'] = float(bmi_match.group(1))
        
    return data
