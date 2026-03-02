import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load the API key from the .env file
load_dotenv()

# 2. Initialize the Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def summarize_text(text):
    print("\n[AI is thinking...]\n")
    
    # 3. Call the model
    # We use gemini-2.5-flash because it is lightning-fast and perfect for text tasks
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction="You are an expert summarizer. Summarize the user's text into exactly 3 concise bullet points. Do not include any intro or outro text.",
            temperature=0.3, # Low temperature = more focused, less hallucination
        )
    )
    
    return response.text

if __name__ == "__main__":
    print("=== The 3-Bullet CLI Summarizer ===")
    print("Paste your long text below.")
    print("When done, press Enter, then Ctrl-D (Mac/Linux) or Ctrl-Z (Windows), then Enter:")
    
    # We use sys.stdin.read() so you can paste multiple paragraphs at once without breaking the script
    user_input = sys.stdin.read()
    
    if user_input.strip():
        summary = summarize_text(user_input)
        print("=== SUMMARY ===")
        print(summary)
        print("\n=================")
    else:
        print("No text provided. Exiting.")