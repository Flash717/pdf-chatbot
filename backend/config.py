"""
Central configuration, read from environment variables with sensible
local-dev defaults. Override any of these with a real .env file or your
process manager / hosting platform's env var settings.
"""
import os
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


BASE_DIR = Path(__file__).resolve().parent.parent

# Where PDFs live and where the vector index is persisted on disk.
PDF_DIR = Path(os.environ.get("PDF_DIR", BASE_DIR / "data" / "pdfs"))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "pdf_docs")

# Persisted BM25 keyword index (see hybrid_search.py), rebuilt after every
# ingestion run and loaded at query time for hybrid search.
BM25_INDEX_PATH = Path(os.environ.get("BM25_INDEX_PATH", BASE_DIR / "bm25_index.pkl"))

# Ollama connection. If Ollama runs on the same machine as this backend,
# the default is correct. If it runs elsewhere, point this at that host.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Model names as known to `ollama pull <name>`. These are common choices
# as of when this was written, but the Ollama model library changes over
# time -- run `ollama list` / check https://ollama.com/library and swap
# these via env vars if a name here is no longer current.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.1")

# Chunking (character-based target size, sentence-based overlap -- see
# chunking.py). Not token-based; simple and good enough for this use case.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
OVERLAP_SENTENCES = int(os.environ.get("OVERLAP_SENTENCES", "2"))

# If true, generate a short LLM-written sentence of context for every
# chunk at ingest time (one extra Ollama chat call per chunk) and prepend
# it to what gets embedded, in addition to the always-on cheap
# doc-name/page-number prefix. Meaningfully improves retrieval on
# multi-document corpora, at the cost of much slower ingestion --
# hundreds of pages means potentially thousands of chunks, i.e. thousands
# of extra local LLM calls. Off by default; turn on if ingest time isn't
# a concern for you.
CONTEXTUALIZE_CHUNKS = _bool_env("CONTEXTUALIZE_CHUNKS", False)

# How many candidates to pull from EACH of vector search and BM25 search
# before fusing and reranking. Should be comfortably larger than TOP_K.
RETRIEVE_CANDIDATES = int(os.environ.get("RETRIEVE_CANDIDATES", "20"))

# Reciprocal Rank Fusion constant (standard default is 60; lower values
# weight top-ranked items more heavily).
RRF_K = int(os.environ.get("RRF_K", "60"))

# Whether to rerank the fused candidate pool with the chat model before
# picking the final TOP_K. Adds one extra local LLM call per question;
# disable if that latency isn't worth it for your setup.
RERANK_ENABLED = _bool_env("RERANK_ENABLED", True)

# How many chunks to actually send to the answering model, after
# retrieval + fusion + (optional) reranking.
TOP_K = int(os.environ.get("TOP_K", "5"))

# How many chunks to send to the embedding model per request during
# ingestion (batching keeps memory/requests reasonable for hundreds of
# pages at once).
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
