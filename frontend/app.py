import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow imports from other folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents import run_rag_chat
from src.vector_store import create_vector_store

# Page configuration
st.set_page_config(
    page_title="MeGPT | Dharmik's Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .stChatMessage {
        background-color: #161b22;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #30363d;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    .profile-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(145deg, #1f2937, #111827);
        border: 1px solid #374151;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Initialization
if 'history' not in st.session_state:
    from langchain_community.chat_message_histories import ChatMessageHistory
    st.session_state.history = ChatMessageHistory()


# Sidebar
with st.sidebar:
    st.markdown("### Contact Details")
    st.markdown("📧 [Email](mailto:dharmikpansuriya107@gmail.com)")
    st.markdown("🐙 [GitHub](https://github.com/dharmik107)")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/dharmik-pansuriya)")

# Main Chat Interface
st.title("🤖 MeGPT Assistant")
st.markdown("Ask anything about Dharmik's education, projects, or background.")

# Display chat history
from langchain_core.messages import HumanMessage
for message in st.session_state.history.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)


# User Input
if prompt := st.chat_input("What would you like to know?"):
    has_api_key = False
    try:
         if "GROQ_API_KEY" in st.secrets or os.environ.get("GROQ_API_KEY"):
             has_api_key = True
    except Exception:
         if os.environ.get("GROQ_API_KEY"):
             has_api_key = True

    if not has_api_key:
        st.error("GROQ_API_KEY is missing! Please configure it in your deployment settings or .env file.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)


        # Generate Assistant response
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            with st.spinner("Agents are retrieving and verifying info..."):
                try:
                    response = run_rag_chat(prompt, st.session_state.history)
                    st.markdown(response)

                except Exception as e:
                    st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Powered by AutoGen, LangChain, FAISS & Groq")
