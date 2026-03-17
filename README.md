# 🤖 MeGPT: Personal AI Assistant

**Live Demo:** [https://megpt-main.streamlit.app](https://megpt-main.streamlit.app)

MeGPT is a **Retrieval-Augmented Generation (RAG)** based AI assistant designed to represent myself. It uses a custom vectorized knowledge base and an agentic verification flow to answer personal, educational, and professional queries with high accuracy and a definitive persona.

---

## ✨ Features

- **Reflective RAG Workflow**: The assistant doesn't just retrieve and generate. It queries the vector database, **verifies** if the retrieved information is sufficient to answer the question, and automatically performs a deeper search if it falls short.
- **Dynamic Age Calculation**: Injects real-time context to ensure the AI can calculate current ages, avoiding "training cutoff" limitations.
- **Premium UI**: Integrated with a clean, dark-themed Streamlit frontend featuring an integrated profile sidebar.
- **Local & Fast**: Uses `LangChain` to orchestrate fast, local FAISS vector embeddings and `ChatGroq` for high-speed LLM processing (`llama-3.1-8b-instant`).

---

## 🏗️ Architecture Stack

- **Large Language Model**: Groq 
- **LLM Orchestration**: LangChain 
- **Vector Database (Embeddings)**: FAISS (`faiss-cpu`) + HuggingFace Models
- **Frontend**: Streamlit
- **Language**: Python 3.10+

---

## 📁 Repository Structure

```text
megpt/
│
├── data/
│   └── about_me.txt         # The core knowledge base describing Myself
├── frontend/
│   └── app.py               # Streamlit UI implementation
├── src/
│   ├── agents.py            # Core RAG verification & generation logic
│   ├── main.py              # Application entry point for Terminal/CLI
│   └── vector_store.py      # Logic for text chunking & FAISS indexing
│
├── .env                     # (Ignored) Stores GROQ API Key
├── requirements.txt         # Project dependencies
├── start.bat                # Windows automation setup launcher
└── README.md                # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.10+ installed. It is highly recommended to use a Conda environment.

```bash
conda create -n megpt python=3.10
conda activate megpt
```

### 2. Install Dependencies
Clone the repository and install the required packages:

```bash
git clone https://github.com/dharmik107/megpt.git
cd megpt
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Launch the AI!
*   **On Windows**: Double-click the `start.bat` file to automatically activate your environment and choose between the Terminal or UI interface.
*   **Via Command Line**:
    *   To run the beautiful UI: `streamlit run frontend/app.py`
    *   To run the terminal version: `python src/main.py`

*(Note: The `faiss_index` database folder will be automatically generated on your first run based on the `data/about_me.txt` file.)*
