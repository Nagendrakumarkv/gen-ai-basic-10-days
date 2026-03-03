import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load the API key
load_dotenv()

# 2. Initialize the modern LangChain Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7, # Higher temperature for more creative video ideas
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# --- CHAIN 1: The Title Generator ---
title_template = """
You are an expert YouTube strategist.
Given the topic: {topic}
Generate exactly 5 highly engaging, click-worthy YouTube video titles.
Return ONLY the titles, one per line.
"""
title_prompt = PromptTemplate.from_template(title_template)

# LCEL: Prompt -> Model -> Clean String Output
title_chain = title_prompt | llm | StrOutputParser()


# --- CHAIN 2: The Script Outline Generator ---
outline_template = """
You are an expert YouTube scriptwriter.
Here are 5 video ideas:
{titles}

Task:
1. Pick the absolute best, most viral title from that list.
2. Write a brief, high-energy 3-part script outline (Hook, Body, Call to Action) for that specific title.
Make sure to state which title you chose at the very top.
"""
outline_prompt = PromptTemplate.from_template(outline_template)

# LCEL: Prompt -> Model -> Clean String Output
outline_chain = outline_prompt | llm | StrOutputParser()


# --- EXECUTE THE PIPELINE ---
if __name__ == "__main__":
    print("=== Automated YouTube Pipeline ===")
    user_topic = input("Enter a YouTube topic (e.g., 'Healthy Cooking', 'Python for Beginners'): ")
    
    print("\n[1/2] Brainstorming 5 viral titles...")
    # We pass the user_topic into the first chain
    generated_titles = title_chain.invoke({"topic": user_topic})
    
    print("\n--- Generated Titles ---")
    print(generated_titles)
    
    print("\n[2/2] Choosing the best title and writing an outline...")
    # We pass the output of Chain 1 (generated_titles) into Chain 2
    script_outline = outline_chain.invoke({"titles": generated_titles})
    
    print("\n--- Final Script Outline ---")
    print(script_outline)
    print("\n===================================")