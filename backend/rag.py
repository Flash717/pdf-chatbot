"""
Retrieval-augmented answer generation: embed the incoming question, pull
the most relevant chunks out of the Chroma index, and ask a local Ollama
chat model to answer using only that retrieved context.
"""
import ollama

import config
from ingest import get_chroma_collection

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context excerpts below, which come from the user's PDF documents. "
    "If the answer is not contained in the context, say plainly that you "
    "don't know rather than guessing or using outside knowledge. When you "
    "use information from the context, mention which source file and "
    "page it came from."
)


def embed_query(question: str, client: ollama.Client = None):
    client = client or ollama.Client(host=config.OLLAMA_HOST)
    response = client.embed(model=config.EMBED_MODEL, input=question)
    return response.embeddings[0]


def retrieve(question: str, top_k: int = None):
    collection = get_chroma_collection()
    query_embedding = embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k or config.TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    hits = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        hits.append({
            "text": doc,
            "source": (meta or {}).get("source"),
            "page": (meta or {}).get("page"),
            "distance": dist,
        })
    return hits


def build_context(hits):
    blocks = [
        f"[Source: {h['source']}, page {h['page']}]\n{h['text']}"
        for h in hits
    ]
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, top_k: int = None):
    hits = retrieve(question, top_k=top_k)
    if not hits:
        return {
            "answer": (
                "I don't have any indexed PDF content to search yet. "
                "Add PDFs to data/pdfs/ and run `python ingest.py` first."
            ),
            "sources": [],
        }

    context = build_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"},
    ]

    client = ollama.Client(host=config.OLLAMA_HOST)
    response = client.chat(model=config.CHAT_MODEL, messages=messages)

    # de-duplicate (source, page) pairs while preserving order
    seen = set()
    sources = []
    for h in hits:
        key = (h["source"], h["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": h["source"], "page": h["page"]})

    return {
        "answer": response.message.content,
        "sources": sources,
    }
