import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def create_vector_store(file_path="data/about_me.txt", db_path="faiss_index"):
    """
    Loads text, creates embeddings, and saves to a FAISS index.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file {file_path} not found.")

    # Load and split documents
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Create embeddings using a reliable HuggingFace model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create and save FAISS index
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(db_path)
    print(f"Vector store created and saved to {db_path}")
    return vector_store

def get_vector_store(db_path="faiss_index"):
    """
    Loads the FAISS index from local storage.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        return None

def query_vector_store(query, db_path="faiss_index", k=2):
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
