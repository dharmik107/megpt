import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_FILE_PATH = os.path.join(BASE_DIR, "data", "about_me.txt")
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "faiss_index")

def create_vector_store(file_path=DEFAULT_FILE_PATH, db_path=DEFAULT_DB_PATH):
    """
    Loads text, creates embeddings, and saves to a FAISS index.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file {file_path} not found.")

    # Load and split documents
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

# Cache embeddings model globally to prevent reloading on every query
_embeddings_instance = None

def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_instance

def create_vector_store(file_path=DEFAULT_FILE_PATH, db_path=DEFAULT_DB_PATH):
    """
    Loads text, creates embeddings, and saves to a FAISS index.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file {file_path} not found.")

    # Load and split documents
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Use cached embeddings
    embeddings = get_embeddings()

    # Create and save FAISS index
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(db_path)
    print(f"Vector store created and saved to {db_path}")
    return vector_store

def get_vector_store(db_path=DEFAULT_DB_PATH):
    """
    Loads the FAISS index from local storage, or creates it if missing.
    """
    embeddings = get_embeddings()
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Index {db_path} not found. Creating it automatically...")
        return create_vector_store(db_path=db_path)

def query_vector_store(query, db_path=DEFAULT_DB_PATH, k=4):
    """
    Queries the vector store for relevant documents.
    """
    vector_store = get_vector_store(db_path)
    if vector_store:
        docs = vector_store.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
    return "No relevant information found."

if __name__ == "__main__":
    # Test creation
    create_vector_store()
