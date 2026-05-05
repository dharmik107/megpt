import os
from dotenv import load_dotenv
from src.vector_store import query_vector_store
from langchain_groq import ChatGroq

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

def get_api_key():
    # Fallback to local .env or environment variables
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
    
    # 1. & 1.5. Reformulate and Classify in ONE call (Major speedup)
    search_query = user_query
    category = "PERSONAL"
    previous_messages = history.messages[-10:] # Last 5 turns approx
    
    # Build history context
    history_text = ""
    if previous_messages:
        for msg in previous_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_text += f"{role}: {msg.content}\n"

    try:
        combined_prompt = f"""Analyze the following conversation history and the new question.
        
        1. **Reformulate**: Rephrase the question to be standalone (resolve pronouns like 'it', 'he'). If independent, keep it as-is.
        2. **Classify**: Is this about Dharmik Pansuriya (skills, projects, background) -> 'PERSONAL' or a general topic -> 'GENERAL'?
        
        History:
        {history_text if history_text else "None"}
        
        New Question: {user_query}
        
        Response format:
        REFORMULATED_QUESTION: [standalone question]
        CATEGORY: [PERSONAL or GENERAL]"""
        
        res = llm.invoke([HumanMessage(content=combined_prompt.strip())])
        lines = res.content.strip().split("\n")
        
        for line in lines:
            if "REFORMULATED_QUESTION:" in line.upper():
                search_query = line.split(":", 1)[1].strip()
            if "CATEGORY:" in line.upper():
                category = line.split(":", 1)[1].strip().upper()
        
        is_personal = "GENERAL" not in category
        print(f"[System] Optimized Routing -> Category: {category}, Query: {search_query}")
        
    except Exception as e:
        print(f"[System Warning] Optimized routing failed: {e}")
        is_personal = True

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
        7. **Accurate Age**: If mentioning his age, ALWAYS calculate it accurately using his Date of Birth (from context) and Today's Date. Do not guess it.

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
