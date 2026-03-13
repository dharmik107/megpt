import os
from dotenv import load_dotenv
from src.vector_store import create_vector_store
from src.agents import run_rag_chat

load_dotenv()

def main():
    print("--- Welcome to MeGPT (Dharmik's Assistant) ---")
    
    # Check if vector store exists, if not create it
    if not os.path.exists("faiss_index"):
        print("Initializing vector database...")
        create_vector_store("data/about_me.txt", "faiss_index")
    
    if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "your_groq_api_key_here":
        print("\n[ERROR] GROQ_API_KEY not found in .env file.")
        print("Please update your .env file with a valid API key.")
        return

    while True:
        query = input("\nAsk something about Dharmik (or type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit', 'bye']:
            break
        
        try:
            run_rag_chat(query)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
