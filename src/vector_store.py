import socket

import socket
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DNS Monkeypatch for restricted environments
if os.environ.get("USE_LOCAL_DNS_PATCH", "").lower() == "true":
    logger.warning("DNS Monkeypatch is ACTIVE. Bypassing DNS for Qdrant.")
    _original_getaddrinfo = socket.getaddrinfo
    def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if host == "b5f4c545-a662-44e9-bcd4-eae1dc547cba.sa-east-1-0.aws.cloud.qdrant.io":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('52.67.45.23', port))]
        return _original_getaddrinfo(host, port, family, type, proto, flags)
    socket.getaddrinfo = _patched_getaddrinfo

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_FILE_PATH = os.path.join(BASE_DIR, "data", "about_me.txt")
COLLECTION_NAME = "megpt_about_me_v2"

_embeddings_instance = None
_qdrant_client_instance = None
_vector_store_instance = None

def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        # FastEmbed is a lightweight, ONNX-based embedding library that does not require PyTorch.
        # This drastically reduces RAM usage (crucial for free tier deployments like Render).
        _embeddings_instance = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings_instance

def get_qdrant_client():
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        _qdrant_client_instance = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY")
        )
    return _qdrant_client_instance

def init_vector_store(file_path=DEFAULT_FILE_PATH):
    """
    Initializes Qdrant connection, checks if collection exists and has data.
    If not, it loads the document and upserts it.
    """
    global _vector_store_instance
    if _vector_store_instance is not None:
        return _vector_store_instance

    client = get_qdrant_client()
    embeddings = get_embeddings()

    # Check if collection exists
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        # Create collection if it doesn't exist. all-MiniLM-L6-v2 has 384 dimensions.
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    
    # Initialize the VectorStore wrapper
    _vector_store_instance = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    # Check if we need to load data
    try:
        count_result = client.count(collection_name=COLLECTION_NAME)
        if count_result.count == 0:
            if not os.path.exists(file_path):
                logger.warning(f"Source file {file_path} not found. Vector DB is empty.")
            else:
                logger.info(f"Index {COLLECTION_NAME} empty. Populating...")
                loader = TextLoader(file_path)
                documents = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)
                _vector_store_instance.add_documents(docs)
                logger.info(f"Vector store populated in collection {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Error checking/populating collection: {e}")

    return _vector_store_instance

def get_vector_store():
    """
    Returns the initialized Qdrant vector store wrapper.
    """
    return init_vector_store()

def query_vector_store(query, k=4):
    """
    Queries the vector store for relevant documents.
    """
    vector_store = get_vector_store()
    if vector_store:
        docs = vector_store.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
    return "No relevant information found."

if __name__ == "__main__":
    # Test creation
    init_vector_store()
