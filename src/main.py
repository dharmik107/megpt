import os
import asyncio
from dotenv import load_dotenv
from src.vector_store import init_vector_store
from src.agents import run_rag_chat_async

load_dotenv()
 
def main():
    print("--- Welcome to MeGPT (Dharmik's Assistant) ---")
    
    # Initialize vector database explicitly
    print("Initializing vector database...")
    init_vector_store("data/about_me.txt")
    
    if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "your_groq_api_key_here":
        print("\n[ERROR] GROQ_API_KEY not found in .env file.")
        print("Please update your .env file with a valid API key.")
        return

    from langchain_community.chat_message_histories import ChatMessageHistory
    history = ChatMessageHistory()

    while True:
        query = input("\nAsk something about Dharmik (or type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit', 'bye']:
            break
        
        try:
            # Pass history to maintain conversation context
            response = asyncio.run(run_rag_chat_async(query, history))
            print(f"\nAI: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
