import os
from dotenv import load_dotenv
from src.vector_store import query_vector_store
from langchain_groq import ChatGroq

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

def get_api_key():
    import streamlit as st
    try:
        # Try Streamlit secrets first (for deployment)
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    
    # Fallback to local .env
    return os.environ.get("GROQ_API_KEY")

# Initialize LangChain ChatGroq as requested by the user
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=get_api_key(),
    temperature=0.0
)

def run_rag_chat(user_query):
    """
    Main entry point for the chat logic. Optimized for speed by removing
    unnecessary verification round-trips.
    """
    # 1. Retrieve Info Directly (Optimized k for reliability)
    print(f"\n[System] Searching for: {user_query}")
    context = query_vector_store(user_query, k=5)
        
    # 2. Generate Final Answer (The Assistant part)
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Using a more direct and concise prompt
    system_message = f"""
    You are MeGPT, the digital assistant for Dharmik Pansuriya. 
    Your goal is to provide **short, on-point, and clean** answers based ONLY on the provided context.

    RULES:
    1. **Be Concise**: Never use 10 words if 5 will do. Avoid fluff, unnecessary introductions, or concluding summaries.
    2. **On-Point**: Answer ONLY what is asked. Do not provide extra information or unsolicited recommendations.
    3. **Format Cleanly**: Use bullet points for lists and bolding for key terms. Keep it readable.
    4. **Source Only**: Use only the provided context. If not found, say you don't know.
    5. **No Meta-Talk**: Do not mention "context", "database", or "cutoffs".
    6. **Persona**: Stay professional and efficient.

    Today's Date: {current_date}
    """

    human_message = f"""
    Context Info:
    ---
    {context}
    ---

    User Question: {user_query}

    Provide a short, direct response:
    """
    
    try:
        # Construct messages for ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_message.strip()),
            HumanMessage(content=human_message.strip())
        ]
        
        res = llm.invoke(messages)
        return res.content.strip()
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"

if __name__ == "__main__":
    # Test
    if get_api_key():
        print(run_rag_chat("What is Dharmik's passion?"))
