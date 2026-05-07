from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from src.agents import run_rag_chat_async
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Vector Store and Models on startup to prevent cold-starts
    from src.vector_store import init_vector_store
    print("[System] Initializing Vector Store on Startup...")
    init_vector_store()
    yield
    print("[System] Shutting down...")

app = FastAPI(title="MeGPT API", lifespan=lifespan)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Reconstruct LangChain history from the request
        history = ChatMessageHistory()
        for msg in request.history:
            if msg.role == "user":
                history.add_user_message(msg.content)
            elif msg.role == "assistant":
                history.add_ai_message(msg.content)
        
        # Run the RAG chat asynchronously
        response = await run_rag_chat_async(request.message, history)
        
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
