"""
Hybrid retrieval support: a BM25 keyword index that complements Chroma's
vector search, plus Reciprocal Rank Fusion (RRF) to merge the two
rankings into one.

Why bother: vector search is good at "meaning" but often misses exact
wording -- names, model numbers, specific dates, acronyms -- because
embeddings compress precise wording into a fuzzier semantic space. BM25
is the opposite: a classic keyword-frequency ranking algorithm (no
embeddings involved) that's very good at exact-term matches but doesn't
understand paraphrase or meaning at all. Combining both catches more of
what either one misses alone.
"""
import pickle
import re

from rank_bm25 import BM25Okapi

import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


def build_bm25_index(collection):
    """Pull every chunk currently in the Chroma collection and build a
    fresh BM25 index over it, persisted to disk so query time doesn't
    have to rebuild it from scratch. Call this after any ingestion run
    (it reflects the *entire* current collection, not just what was just
    added, since BM25's scoring statistics depend on the whole corpus).
    """
    data = collection.get(include=["documents", "metadatas"])
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    config.BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not ids:
        if config.BM25_INDEX_PATH.exists():
            config.BM25_INDEX_PATH.unlink()
        return None

    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    index = {"bm25": bm25, "ids": ids, "documents": documents, "metadatas": metadatas}
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    return index


_cached_index = None
_cached_mtime = None


def load_bm25_index(force_reload: bool = False):
    """Load the persisted BM25 index, caching it in memory for the life
    of the process and transparently reloading if the file on disk has
    changed (e.g. ingest.py was re-run while the API server stayed up).
    Returns None if no index has been built yet.
    """
    global _cached_index, _cached_mtime

    if not config.BM25_INDEX_PATH.exists():
        return None

    mtime = config.BM25_INDEX_PATH.stat().st_mtime
    if force_reload or _cached_index is None or mtime != _cached_mtime:
        with open(config.BM25_INDEX_PATH, "rb") as f:
            _cached_index = pickle.load(f)
        _cached_mtime = mtime

    return _cached_index


def bm25_search(query: str, top_n: int):
    """Return up to top_n {id, text, source, page, score} dicts ranked by
    BM25 score, highest first. Empty list if no index exists yet or the
    query has no recognizable tokens (e.g. punctuation-only)."""
    index = load_bm25_index()
    if index is None:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = index["bm25"].get_scores(tokenized_query)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked[:top_n]:
        if scores[i] <= 0:
            continue  # no keyword overlap at all -- not a real match
        meta = index["metadatas"][i] or {}
        results.append({
            "id": index["ids"][i],
            "text": index["documents"][i],
            "source": meta.get("source"),
            "page": meta.get("page"),
            "score": float(scores[i]),
        })
    return results


def reciprocal_rank_fusion(ranked_lists, k: int = 60):
    """Merge multiple ranked lists of items (each item a dict with an
    "id" key) into one fused ranking. Each item's fused score is the sum
    of 1/(k + rank) across every list it appears in, where rank is its
    0-indexed position in that list. An item found by both searches
    naturally outranks one found by only one.

    RRF needs no score normalization between BM25 scores and vector
    distances -- which live on completely different, incomparable scales
    -- which is exactly why it's the standard way to combine rankers
    like this rather than, say, averaging raw scores.
    """
    fused_scores = {}
    item_by_id = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            item_id = item["id"]
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + 1.0 / (k + rank)
            item_by_id.setdefault(item_id, item)

    ordered_ids = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)
    return [dict(item_by_id[i], fused_score=fused_scores[i]) for i in ordered_ids]
