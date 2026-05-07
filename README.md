# 🌟 MeGPT - Your Personalized Digital Assistant

[![Demo Link](https://img.shields.io/badge/Live_Demo-MeGPT-58a6ff?style=for-the-badge&logo=vercel)](https://megpt-ten.vercel.app/)

MeGPT is an advanced **Agentic AI Digital Clone** built to answer questions about Dharmik Pansuriya's professional and personal profile. It leverages Retrieval-Augmented Generation (RAG) and conversational memory to provide accurate, concise, and contextual responses based strictly on Dharmik's real data.

👉 **Try it out here:** [https://megpt-ten.vercel.app/](https://megpt-ten.vercel.app/)

---

## 🛠️ Modern Tech Stack

This project recently underwent a massive modernization to ensure high performance and seamless cloud deployment.

- **Frontend**: React (Vite) for a fast, responsive, and aesthetically pleasing Glassmorphism UI.
- **Backend API**: FastAPI for blazing-fast orchestration and concurrent request handling.
- **LLM Engine**: Groq (`llama-3.1-8b-instant`) for ultra low-latency inference.
- **Orchestration**: LangChain & LangChain-Community.
- **Embeddings**: FastEmbed (`BAAI/bge-small-en-v1.5`) – A lightweight, ONNX-based embedding model replacing heavy PyTorch workflows to easily fit within cloud memory limits.
- **Vector Database**: Qdrant Cloud (formerly local FAISS) for scalable and durable vector storage.
- **Deployment**: Vercel (Frontend) & Render (Backend).

---

## ⚡ High-Performance RAG Architecture

The backend implements a highly optimized, low-latency conversational pipeline:

1. **Intelligent Query Resolution**: For follow-up questions, the AI analyzes the last 5 turns of conversation to resolve pronouns (e.g., turning *"What is it?"* into *"What is the HireNova project?"*). *Optimization: This step is automatically bypassed on the very first message to cut latency in half.*
2. **Lightning-fast Vector Retrieval**: The query is embedded locally via FastEmbed and sent to Qdrant Cloud to retrieve relevant context (~100ms).
3. **Dynamic Generation**: The context and chat history are passed to Groq. The AI is specifically instructed to use the database context for personal questions, but fallback to general knowledge if the user asks a non-personal question.

---

## 📂 Project Structure

```bash
MeGPT/
│
├── data/               # Raw profile definitions (.txt)
├── frontend/           # React Application (Vite + Tailwind/CSS)
├── src/                
│   ├── agents.py       # Core LangChain RAG pipeline & LLM generation
│   ├── api.py          # FastAPI server, endpoints, and CORS
│   ├── main.py         # Standalone CLI testing script
│   └── vector_store.py # FastEmbed & Qdrant integration
├── .env                # Environment overrides (API keys)
├── Procfile            # Deployment configuration for Render
└── requirements.txt    # Frozen Python dependencies
```

---

## 🚀 Running Locally

### 1. Backend Setup
1. Clone the repository.
2. Create an environment and install dependencies: `pip install -r requirements.txt`
3. Set your API keys in the `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   USE_LOCAL_DNS_PATCH=true # If required by your local network
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

### 2. Frontend Setup
1. Navigate to the frontend directory: `cd frontend`
2. Install dependencies: `npm install`
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Open `http://localhost:5173/` in your browser.
