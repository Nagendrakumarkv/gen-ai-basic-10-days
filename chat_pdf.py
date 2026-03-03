import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load your API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("[1/4] Loading PDF...")
# 2. Load the PDF
# Make sure you have a file named 'document.pdf' in your folder!
loader = PyPDFLoader("document.pdf")
docs = loader.load()

print("[2/4] Splitting text into chunks...")
# 3. Split the document into chunks
# Overlap ensures sentences at the edge of a chunk don't lose context
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print("[3/4] Creating Embeddings and saving to Vector Database...")
# 4. Initialize Google's Embedding Model and Chroma Vector DB
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 5. Create a Retriever (the search engine for our DB)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 closest chunks

# 6. Set up the LLM and the Prompt
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)

template = """
You are a helpful assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

# --- THE RAG PIPELINE (LCEL) ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# This pipe routes the question to the retriever, formats the docs, passes everything to the prompt, and generates the answer.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("[4/4] System Ready!\n")

# --- INTERACTIVE CHAT LOOP ---
if __name__ == "__main__":
    print("=== Chat with your PDF ===")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_question = input("Ask a question about the PDF: ")
        
        if user_question.lower() in ['exit', 'quit']:
            break
            
        print("\nSearching and thinking...")
        response = rag_chain.invoke(user_question)
        print("\n--- Answer ---")
        print(response)
        print("----------------\n")