"""
Central configuration, read from environment variables with sensible
local-dev defaults. Override any of these with a real .env file or your
process manager / hosting platform's env var settings.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where PDFs live and where the vector index is persisted on disk.
PDF_DIR = Path(os.environ.get("PDF_DIR", BASE_DIR / "data" / "pdfs"))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "pdf_docs")

# Ollama connection. If Ollama runs on the same machine as this backend,
# the default is correct. If it runs elsewhere, point this at that host.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Model names as known to `ollama pull <name>`. These are common choices
# as of when this was written, but the Ollama model library changes over
# time -- run `ollama list` / check https://ollama.com/library and swap
# these via env vars if a name here is no longer current.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.1")

# Chunking (character-based, not token-based -- simple and good enough
# for this use case).
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# How many chunks to retrieve per question.
TOP_K = int(os.environ.get("TOP_K", "5"))

# How many chunks to send to the embedding model per request during
# ingestion (batching keeps memory/requests reasonable for hundreds of
# pages at once).
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
