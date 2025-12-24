import os
from openai import OpenAI
from dotenv import load_dotenv

def test_connection():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not found in .env file.")
        print("Please ensure you have created a .env file with format: OPENAI_API_KEY=sk-...")
        return

    # Mask key for display
    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
    print(f"[INFO] Found API Key: {masked_key}")

    try:
        print("[INFO] Attempting to connect to OpenAI API (Model: gpt-4o-mini)...")
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with 'API Working'"}]
        )
        
        reply = response.choices[0].message.content
        if reply:
            print(f"\n[SUCCESS] Connection Established! Model replied:\n> {reply}")
        else:
            print("\n[WARNING] Connected, but received empty response.")
            
    except Exception as e:
        print(f"\n[FAILURE] Connection Failed.\nError: {e}")

if __name__ == "__main__":
    test_connection()
