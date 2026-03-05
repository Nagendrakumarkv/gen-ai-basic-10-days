import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# 1. Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Initialize the LLM (Temperature 0.7 for empathy and conversational variance)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=api_key)

# 3. Create the Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Dr. AI, a highly empathetic, insightful personal therapist. "
               "Listen to the user, validate their feelings, and ask gentle follow-up questions to help them reflect. "
               "Keep your responses concise and conversational (1-3 sentences max)."),
    # This placeholder is where all previous messages will be injected!
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 4. Create the basic chain using LCEL
chain = prompt | llm

# 5. Set up the Memory Store (A dictionary to hold different user sessions)
store = {}

# This function retrieves or creates a blank memory buffer for a specific session ID
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 6. Wrap the chain with the History Manager
therapist_bot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# --- EXECUTE THE BOT ---
if __name__ == "__main__":
    print("=== Dr. AI: Personal Therapist ===")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    # We define a hardcoded session ID for this single terminal user.
    # In a real web app, this would be the user's logged-in ID.
    session_config = {"configurable": {"session_id": "user_123"}}
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Dr. AI: Take care of yourself. I'll be here if you need me.")
            break
            
        # We pass both the input AND the config containing the session_id
        response = therapist_bot.invoke(
            {"input": user_input},
            config=session_config
        )
        
        print(f"Dr. AI: {response.content}\n")