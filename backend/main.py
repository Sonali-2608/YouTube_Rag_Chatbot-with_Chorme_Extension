from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag_utils import run_rag


app = FastAPI(title="YouTube RAG Chatbot")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        
    allow_credentials=True,
    allow_methods=["*"],       
    allow_headers=["*"],        
)


class RAGRequest(BaseModel):
    video_id: str
    question: str


@app.get("/")
def read_root():
    return {"message": "Welcome to YouTube RAG Chatbot API"}


@app.post("/chat")
def chat(request: RAGRequest):
    try:
        start_time = perf_counter()
        answer = run_rag(request.video_id, request.question)
        latency_seconds = perf_counter() - start_time
        print(f"Chat latency: {latency_seconds:.2f}s")
        return {"answer": answer, "latency_seconds": round(latency_seconds, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
