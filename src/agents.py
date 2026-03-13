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

def retrieve_and_verify(user_query):
    """
    Agentic RAG Logic:
    Retrieves info -> Verifies relevance (using ChatGroq) -> Re-retrieves if necessary.
    """
    print(f"\n[System] Searching for: {user_query}")
    # Increased initial retrieval depth to ensure we capture spread-out lists (like projects)
    retrieved_info = query_vector_store(user_query, k=5)
    
    # Verification prompt
    verification_prompt = f"""
    You are a strict data verifier for Dharmik's Assistant.
    
    User Query: {user_query}
    Retrieved Information from Database: {retrieved_info}
    
    Task: Is the retrieved information relevant and sufficient to answer the User Query accurately?
    Respond with ONLY 'YES' or 'NO'.
    """
    
    try:
        response = llm.invoke(verification_prompt)
        is_correct = response.content.strip().upper() == "YES"
    except Exception as e:
        print(f"[Warning] Verification failed: {e}. Defaulting to YES.")
        is_correct = True
    
    if not is_correct:
        print("[System] Information was not sufficient. Attempting deeper retrieval...")
        # Deep search to pull a very broad context if the initial one failed
        retrieved_info = query_vector_store(user_query, k=10)
    
    return retrieved_info

def run_rag_chat(user_query):
    """
    Main entry point for the agentic chat logic.
    """
    # 1. Retrieve and Verify (The Agentic part)
    context = retrieve_and_verify(user_query)
        
    # 2. Generate Final Answer (The Assistant part)
    # Inject current date context to prevent "knowledge cutoff" explanations
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    final_prompt = f"""
    You are MeGPT, the personal assistant representing Dharmik Pansuriya.
    Current Date: {current_date}
    
    INSTRUCTIONS:
    - Answer the User Question using ONLY the provided Context.
    - Be extremely thorough: If the user asks for a list (like projects or skills), extract and list EVERYTHING mentioned in the context. Do not stop at just one item.
    - Be concise, direct, and professional in your delivery. Let the data speak for itself.
    - If asked about age, calculate it accurately using the Date of Birth in the context and today's date ({current_date}). Format it clearly (e.g., "As the date of birth is August 15, 2005, the age of Dharmik is 21.")
    - Do NOT mention "knowledge cutoffs", "databases", or "retrieved information" to the user.
    - If the context doesn't contain the answer at all, politely state you don't have that specific information about Dharmik.
    
    Context:
    {context}
    
    User Question: {user_query}
    
    Assistant Response:
    """
    
    try:
        res = llm.invoke(final_prompt)
        return res.content.strip()
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"

if __name__ == "__main__":
    # Test
    if get_api_key():
        print(run_rag_chat("What is Dharmik's passion?"))
