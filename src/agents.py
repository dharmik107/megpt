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

def run_rag_chat(user_query, history=None):
    """
    Main entry point for the chat logic. Optimized for speed and supports memory.
    """
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    
    # Initialize history if not provided
    if history is None:
        history = ChatMessageHistory()

    print(f"\n[System] Original Query: {user_query}")
    
    # 1. Reformulate Query if history exists to resolve pronouns (e.g., "it")
    search_query = user_query
    previous_messages = history.messages[-10:] # Last 5 turns approx
    
    if previous_messages:
        try:
            # Build history text for reformulation
            history_text = ""
            for msg in previous_messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_text += f"{role}: {msg.content}\n"
            
            reformulate_prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a **standalone question**.

RULES:
1. **Pronoun Resolution**: If the question uses pronouns ("it", "he", "she", "that", "the project"), replace them with the actual subject from the history.
2. **Independence**: If the question is independent of the history (e.g., general knowledge, math, cities), return it **EXACTLY as-is**. Do NOT force a connection to the history or append "in context".
3. **No Answers**: Do NOT answer the question, just rephrase it.

History:
{history_text}

Follow-up Question: {user_query}
Standalone Question:"""
            
            # Use same LLM for fast reformulation
            reform_res = llm.invoke([HumanMessage(content=reformulate_prompt.strip())])
            search_query = reform_res.content.strip().split("\n")[0].strip() # Take first line to be safe
            print(f"[System] Reformulated Search Query: {search_query}")
        except Exception as e:
            print(f"[System Warning] Failed to reformulate query: {e}")
            search_query = user_query

    # 1.5. Classify Query (Personal vs General)
    is_personal = True
    try:
        classification_prompt = f"""Classify the following question as:
- 'PERSONAL': About Dharmik Pansuriya (skills, projects, background, salary expectations, preferences, job requirements).
- 'GENERAL': Any other general topic (geography, math, generic coding, common facts).

Response MUST be just one word: PERSONAL or GENERAL.

Question: {search_query}
Category:"""
        class_res = llm.invoke([HumanMessage(content=classification_prompt.strip())])
        category = class_res.content.strip().upper()
        # Flexibly match to avoid strict formatting index issues
        is_personal = "GENERAL" not in category or \
                      any(kw in search_query.lower().split() for kw in ["dharmik", "his", "he", "your"]) or \
                      any(term in search_query.lower() for term in ["hirenova", "repopilot", "megpt"])
        print(f"[System] Category: {'PERSONAL' if is_personal else 'GENERAL'}")
    except Exception as e:
        print(f"[System Warning] Classification failed: {e}")

    # 2. Retrieve Info (Conditional)
    context = ""
    if is_personal:
        print(f"[System] Searching Vector DB for: {search_query}")
        context = query_vector_store(search_query, k=5)
    else:
        print(f"[System] General Knowledge Query - Bypassing vector search.")

    # 3. Generate Final Answer
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    if is_personal:
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

        User Question: {search_query}

        Provide a short, direct response:
        """
    else:
        system_message = f"""
        You are MeGPT, a helpful and efficient AI assistant. 
        Answer the question using your general knowledge accurately and concisely.

        Today's Date: {current_date}
        """

        human_message = f"""
        User Question: {search_query}

        Provide a helpful and direct response:
        """
    
    try:
        # Construct messages for ChatGroq including history
        messages = [SystemMessage(content=system_message.strip())]
        
        # Add limited history (last 10 messages = 5 turns)
        messages.extend(previous_messages)
        
        # Add current message
        messages.append(HumanMessage(content=human_message.strip()))
        
        res = llm.invoke(messages)
        response_content = res.content.strip()
        
        # Save current turn to history
        history.add_user_message(user_query)
        history.add_ai_message(response_content)
        
        return response_content
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"



if __name__ == "__main__":
    # Test
    if get_api_key():
        print(run_rag_chat("What is Dharmik's passion?"))
