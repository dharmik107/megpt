import os
import asyncio
import logging
from dotenv import load_dotenv
from src.vector_store import query_vector_store
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

async def run_rag_chat_async(user_query, history=None):
    """
    Main entry point for the chat logic. Optimized for speed and supports memory.
    """
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    
    # Initialize history if not provided
    if history is None:
        history = ChatMessageHistory()

    logger.info(f"Original Query: {user_query}")
    
    previous_messages = history.messages[-10:] # Last 5 turns approx
    search_query = user_query
    
    # 1. Reformulate ONLY if there is history
    if previous_messages:
        # Build history context
        history_text = ""
        for msg in previous_messages[-4:]: # Use last 2 turns for context to save tokens
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_text += f"{role}: {msg.content}\n"

        try:
            reformulate_prompt = f"""Analyze the conversation history and rewrite the new question to be a standalone question (resolve pronouns like 'it', 'he'). If it is already independent, keep it as-is. DO NOT answer the question, just output the reformulated question.
            
            History:
            {history_text}
            
            New Question: {user_query}
            
            Reformulated Question:"""
            
            # Async call to prevent blocking event loop
            res = await llm.ainvoke([HumanMessage(content=reformulate_prompt.strip())])
            search_query = res.content.strip()
            
            # Clean up the output in case the LLM was chatty
            if ":" in search_query and len(search_query.split(":")[0]) < 25:
                search_query = search_query.split(":", 1)[1].strip()
                
            logger.info(f"Reformulated Query: {search_query}")
            
        except Exception as e:
            logger.warning(f"Reformulation failed: {e}")

    # 2. Retrieve Info ALWAYS
    logger.info(f"Searching Vector DB for: {search_query}")
    # Run synchronous vector db call in a thread to avoid blocking event loop
    context = await asyncio.to_thread(query_vector_store, search_query, k=5)

    # 3. Generate Final Answer
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    system_message = f"""
    You are MeGPT, the digital assistant for Dharmik Pansuriya. 
    Your goal is to provide **short, on-point, and clean** answers.

    RULES:
    1. **Be Concise**: Never use 10 words if 5 will do. Avoid fluff, unnecessary introductions, or concluding summaries.
    2. **On-Point**: Answer ONLY what is asked. Do not provide extra information or unsolicited recommendations.
    3. **Format Cleanly**: Use bullet points for lists and bolding for key terms. Keep it readable.
    4. **Context Usage**: Use the provided context IF it is relevant to the user's question about Dharmik. If the question is a general knowledge question (e.g., 'what is python?'), ignore the context and answer normally using your general knowledge.
    5. **No Meta-Talk**: Do not mention "context", "database", or "cutoffs". If you don't know something about Dharmik, just say you don't know.
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
    
    try:
        # Construct messages for ChatGroq including history
        messages = [SystemMessage(content=system_message.strip())]
        
        # Add limited history (last 10 messages = 5 turns)
        messages.extend(previous_messages)
        
        # Add current message
        messages.append(HumanMessage(content=human_message.strip()))
        
        # Async call to prevent blocking event loop
        res = await llm.ainvoke(messages)
        response_content = res.content.strip()
        
        # Save current turn to history
        history.add_user_message(user_query)
        history.add_ai_message(response_content)
        
        return response_content
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        return f"An error occurred while generating the answer: {e}"

if __name__ == "__main__":
    # Test
    if get_api_key():
        print(asyncio.run(run_rag_chat_async("What is Dharmik's passion?")))
