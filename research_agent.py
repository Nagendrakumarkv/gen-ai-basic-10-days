import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# 1. Load API Keys
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# 2. Initialize the LLM
# We use temperature 0 because we want factual financial data, not creative fiction.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=gemini_key)

# 3. Equip the Agent with Tools
# We give it the Tavily search tool, configured to return the top 3 results.
search_tool = TavilySearchResults(max_results=3, tavily_api_key=tavily_key)
tools = [search_tool]

# 4. Compile the Agent using LangGraph
# This creates the ReAct loop automatically.
agent_executor = create_react_agent(llm, tools)

if __name__ == "__main__":
    print("=== Autonomous Market Research Agent ===")
    
    # You can change this prompt to anything!
    user_query = "What is the current price of Apple (AAPL) stock, and what are the top 2 recent news headlines about the company? Based on this, summarize the market sentiment."
    
    print(f"\nUser: {user_query}\n")
    print("[Agent is thinking and browsing the web...]\n")
    
    # 5. Run the Agent
    # We pass the input as a HumanMessage
    response = agent_executor.invoke({"messages": [HumanMessage(content=user_query)]})
    
    # 6. Extract and print the final answer
    # The response dictionary contains the entire history of the agent's thoughts and actions.
    # We just want the very last message it generated.
    final_answer = response["messages"][-1].content
    
    print("=== Final Report ===")
    print(final_answer)
    print("====================")