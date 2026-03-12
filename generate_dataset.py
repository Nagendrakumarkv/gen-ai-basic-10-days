import os
import json
from dotenv import load_dotenv
from google import genai

# 1. Load API Key
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def create_training_data():
    print("=== JSONL Training Data Generator ===")
    print("Generating 3 synthetic examples... (In reality, you'd loop this to make 100+)\n")
    
    # We ask Gemini to invent customer queries and sarcastic responses
    prompt = """
    Generate 3 distinct examples of a user asking a generic customer service question, 
    and a highly sarcastic, dry response from the agent.
    
    Output ONLY valid JSON in this exact format, with no markdown formatting or code blocks:
    [
      {"user": "question here", "agent": "sarcastic response here"}
    ]
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    
    # Clean the output and parse the JSON
    raw_text = response.text.replace('```json', '').replace('```', '').strip()
    
    try:
        synthetic_examples = json.loads(raw_text)
    except json.JSONDecodeError:
        print("❌ Error: AI didn't return perfect JSON. Try running it again.")
        return
        
    # --- WRITE TO JSONL ---
    output_filename = "sarcastic_bot_training.jsonl"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        for example in synthetic_examples:
            # We map the AI's generation into the strict fine-tuning format
            training_row = {
                "messages": [
                    {"role": "system", "content": "You are a sarcastic customer support agent."},
                    {"role": "user", "content": example["user"]},
                    {"role": "model", "content": example["agent"]}
                ]
            }
            # Write exactly one JSON object per line, followed by a newline
            f.write(json.dumps(training_row) + "\n")
            
    print(f"✅ Successfully created '{output_filename}'!")
    print("Open the file in your code editor to see the exact format required for Fine-Tuning.")

if __name__ == "__main__":
    create_training_data()