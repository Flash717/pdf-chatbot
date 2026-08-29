"""
FastAPI backend for the PDF chatbot.

Endpoints:
  GET  /health              -> basic status check
  POST /chat                -> { "question": "..." } -> { "answer", "sources" }
  POST /ingest               -> multipart PDF upload(s); ingests them immediately

Run locally:
    uvicorn main:app --reload --port 8000

Requires Ollama running separately (`ollama serve`) with the configured
models pulled (see README).
"""
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from ingest import get_chroma_collection, ingest_pdf
from rag import answer_question

app = FastAPI(title="PDF Chatbot API")

# Wide open for local development. Before deploying behind your website,
# replace "*" with your site's actual origin(s), e.g. ["https://example.com"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class SourceRef(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceRef]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        return answer_question(req.question, top_k=req.top_k)
    except Exception as e:
        # Most common cause locally: Ollama isn't running, or the model
        # isn't pulled yet.
        raise HTTPException(status_code=502, detail=f"generation failed: {e}")


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    collection = get_chroma_collection()

    results = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            results.append({"file": f.filename, "status": "skipped (not a .pdf)"})
            continue
        dest = config.PDF_DIR / f.filename
        dest.write_bytes(await f.read())
        try:
            n_chunks = ingest_pdf(dest, collection)
            results.append({"file": f.filename, "chunks_added": n_chunks})
        except Exception as e:
            results.append({"file": f.filename, "status": f"failed: {e}"})

    return {"results": results}
