import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# 1. UI Setup
st.set_page_config(page_title="Resume Roaster", page_icon="🔥", layout="wide")
st.title("🔥 The GenAI Resume Roaster & Fixer")
st.markdown("Upload your resume and paste a Job Description URL. The AI will roast your current resume and rewrite it to get you the interview.")

# 2. Load API Key (Optimized for Production)
load_dotenv()
# Try to get it from Streamlit Secrets (Cloud), otherwise use local .env
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("🚨 Missing Gemini API Key. If running locally, check your .env file. If deployed, check Streamlit Secrets.")
    st.stop()

# 3. Initialize the LLM
# We use temperature 0.4: low enough to be accurate to the resume, high enough to be creative with the roast.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, google_api_key=api_key)

# 4. The Engineering Prompt
template = """
You are an elite, brutally honest tech recruiter and career coach. 
Your job is to analyze the provided Resume against the provided Job Description.

Do not hold back. Be witty, direct, and highly actionable. 

Format your response exactly like this:

## 🎯 ATS Match Score: [Insert Score 0-100]%

## 🔥 The Roast
[Write 2-3 paragraphs roasting the resume. Point out exactly why this resume would get thrown in the trash for this specific job. Highlight missing keywords, weak verbs, and fluff.]

## 🛠️ The Fixes (Rewritten Bullets)
[Rewrite the 3 weakest bullet points in the resume so they perfectly align with the job description. Use the 'Accomplished [X] as measured by [Y], by doing [Z]' framework.]

---
**Job Description:**
{job_description}

---
**Resume:**
{resume_text}
"""
prompt = PromptTemplate.from_template(template)
roast_chain = prompt | llm

# 5. Sidebar Inputs
with st.sidebar:
    st.header("Your Details")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_url = st.text_input("Job Description URL")
    analyze_button = st.button("Roast My Resume", type="primary")

# 6. Execution Logic
if analyze_button:
    if not uploaded_file or not job_url:
        st.warning("⚠️ Please provide both a PDF resume and a Job URL.")
    else:
        with st.spinner("Scraping job description and reading your life's work..."):
            try:
                # --- A. Scrape the Job Description ---
                # Note: Some highly secure sites (like LinkedIn) block basic scrapers. 
                # For this MVP, test it with a standard company careers page or Indeed.
                loader_web = WebBaseLoader(job_url)
                jd_docs = loader_web.load()
                job_text = jd_docs[0].page_content

                # --- B. Read the PDF Resume ---
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                loader_pdf = PyPDFLoader(tmp_file_path)
                resume_docs = loader_pdf.load()
                
                # Combine PDF pages into one giant string (Context Stuffing)
                resume_text = "\n".join([doc.page_content for doc in resume_docs])
                
                # --- C. Run the Chain ---
                response = roast_chain.invoke({
                    "job_description": job_text,
                    "resume_text": resume_text
                })
                
                # Display the output
                st.markdown(response.content)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Tip: If the error is about scraping, the website might be blocking bots. Try a different URL!")