import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load your Gemini API Key
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_caption():
    print("=== Auto-Instagram Captioner (Local File) ===")
    
    # 2. Check if the image exists
    image_path = "image.jpg"
    if not os.path.exists(image_path):
        print(f"\n❌ Error: I couldn't find a file named '{image_path}'.")
        print("Please save an image in this folder as 'image.jpg' and try again.")
        return

    print("\n[1/1] Uploading image and writing caption...")
    
    # 3. Read the local image file into bytes
    with open(image_path, "rb") as f:
        raw_image_bytes = f.read()
    
    # 4. Package it for Gemini
    image_part = types.Part.from_bytes(data=raw_image_bytes, mime_type="image/jpeg")
    
    # 5. Call the Vision model
    caption_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            image_part, 
            "You are a viral social media manager. Look at this image and write a short, witty, and highly engaging Instagram caption for it. Include exactly 3 relevant hashtags at the end."
        ],
    )
    
    print("\n=== YOUR INSTAGRAM POST ===")
    print(caption_response.text)
    print("===========================\n")

if __name__ == "__main__":
    generate_caption()