"""
Retrieval-augmented answer generation.

Pipeline per question:
  1. Vector search (Chroma) and BM25 keyword search each retrieve
     RETRIEVE_CANDIDATES chunks (hybrid_search.py) -- vector search
     catches semantic/paraphrased matches, BM25 catches exact terms
     (names, numbers, acronyms) that embeddings sometimes blur.
  2. The two ranked lists are merged with Reciprocal Rank Fusion.
  3. If RERANK_ENABLED, the fused candidate pool is reranked by asking
     the local chat model to judge relevance directly -- generally more
     accurate than either search's own ranking, at the cost of one more
     local LLM call per question.
  4. The final TOP_K chunks are sent to the chat model as context to
     answer the question, with source citations.
"""
import json

import ollama

import config
from hybrid_search import bm25_search, reciprocal_rank_fusion
from ingest import get_chroma_collection

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context excerpts below, which come from the user's PDF documents. "
    "If the answer is not contained in the context, say plainly that you "
    "don't know rather than guessing or using outside knowledge. When you "
    "use information from the context, mention which source file and "
    "page it came from."
)

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "Candidate numbers (the [N] labels), ordered from most to "
                "least relevant to answering the question. Include every "
                "candidate number exactly once."
            ),
        }
    },
    "required": ["ranking"],
}


def embed_query(question: str, client: ollama.Client = None):
    client = client or ollama.Client(host=config.OLLAMA_HOST)
    response = client.embed(model=config.EMBED_MODEL, input=question)
    return response.embeddings[0]


def vector_search(question: str, top_n: int, client: ollama.Client = None):
    """Chroma vector similarity search. Returns hits with an "id" field so
    they can be fused with BM25 results by id."""
    collection = get_chroma_collection()
    query_embedding = embed_query(question, client)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )

    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    hits = []
    for id_, doc, meta, dist in zip(ids, documents, metadatas, distances):
        meta = meta or {}
        hits.append({
            "id": id_,
            "text": doc,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "distance": dist,
        })
    return hits


def hybrid_retrieve(question: str, candidate_pool: int = None):
    """Run vector + BM25 search and fuse them into one ranked candidate
    pool via Reciprocal Rank Fusion."""
    candidate_pool = candidate_pool or config.RETRIEVE_CANDIDATES
    client = ollama.Client(host=config.OLLAMA_HOST)

    vector_hits = vector_search(question, candidate_pool, client)
    keyword_hits = bm25_search(question, candidate_pool)

    fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=config.RRF_K)
    return fused[:candidate_pool]


def rerank(question: str, candidates, top_k: int, client: ollama.Client = None):
    """Ask the chat model to rank the fused candidates by relevance and
    keep the best top_k. Falls back to the incoming (fused) order if the
    model call fails or returns something we can't parse -- reranking is
    a quality improvement, not something that should break answering."""
    if not candidates:
        return []
    if not config.RERANK_ENABLED or len(candidates) <= top_k:
        return candidates[:top_k]

    client = client or ollama.Client(host=config.OLLAMA_HOST)

    numbered = "\n\n".join(
        f"[{i}] (source: {c['source']}, page {c['page']})\n{c['text']}"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"Question: {question}\n\n"
        "Below are numbered candidate excerpts from a document collection. "
        "Rank ALL of them from most to least relevant to answering the "
        "question above.\n\n" + numbered
    )

    try:
        response = client.chat(
            model=config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format=RERANK_SCHEMA,
        )
        parsed = json.loads(response.message.content)
        raw_ranking = parsed.get("ranking", [])

        seen = set()
        ordered_indices = []
        for i in raw_ranking:
            if isinstance(i, int) and 0 <= i < len(candidates) and i not in seen:
                seen.add(i)
                ordered_indices.append(i)
        # Anything the model omitted keeps its original fused-order position
        # at the end, rather than being silently dropped.
        for i in range(len(candidates)):
            if i not in seen:
                ordered_indices.append(i)

        return [candidates[i] for i in ordered_indices][:top_k]
    except Exception as e:
        print(f"[!] Rerank failed, falling back to fused order: {e}")
        return candidates[:top_k]


def build_context(hits):
    blocks = [
        f"[Source: {h['source']}, page {h['page']}]\n{h['text']}"
        for h in hits
    ]
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, top_k: int = None):
    top_k = top_k or config.TOP_K
    candidates = hybrid_retrieve(question)

    if not candidates:
        return {
            "answer": (
                "I don't have any indexed PDF content to search yet. "
                "Add PDFs to data/pdfs/ and run `python ingest.py` first."
            ),
            "sources": [],
        }

    final_hits = rerank(question, candidates, top_k)

    context = build_context(final_hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"},
    ]

    client = ollama.Client(host=config.OLLAMA_HOST)
    response = client.chat(model=config.CHAT_MODEL, messages=messages)

    seen = set()
    sources = []
    for h in final_hits:
        key = (h["source"], h["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": h["source"], "page": h["page"]})

    return {
        "answer": response.message.content,
        "sources": sources,
    }
