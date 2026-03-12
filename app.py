import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. UI Setup (This must be the very first Streamlit command)
st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with your private PDF")

# 2. Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Missing Gemini API Key. Please check your .env file.")
    st.stop()

# 3. Initialize Session State (Memory)
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores the chat history

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None # Stores the LangChain pipeline so we don't rebuild it

# 4. The Sidebar (File Upload & Processing)
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF to analyze", type=["pdf"])
    
    # If a file is uploaded AND we haven't processed it yet...
    if uploaded_file and st.session_state.rag_chain is None:
        with st.spinner("Crunching the document... This takes a few seconds."):
            
            # Save the uploaded file temporarily so LangChain can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                
            # --- THE DAY 3 RAG LOGIC ---
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
            
            template = """
            You are a helpful assistant. Use ONLY the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            
            Context: {context}
            Question: {question}
            Answer:
            """
            prompt = PromptTemplate.from_template(template)
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
                
            # Save the fully built chain into Session State!
            st.session_state.rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            st.success("PDF processed! You can now chat.")

# 5. The Main Chat Interface
# Redraw all past messages when the script reruns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. The User Input
if user_input := st.chat_input("Ask a question about your PDF..."):
    
    # Don't let them ask questions if they haven't uploaded a PDF
    if st.session_state.rag_chain is None:
        st.error("Please upload a PDF in the sidebar first!")
    else:
        # Show the user's message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Generate the AI's response using the RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                response = st.session_state.rag_chain.invoke(user_input)
                st.markdown(response)
        
        # Save the AI's response to history
        st.session_state.messages.append({"role": "assistant", "content": response})