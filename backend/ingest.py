"""
Extract text from every PDF in PDF_DIR, split it into paragraph/sentence-
aware chunks, embed each chunk (with a bit of prepended context) using a
local Ollama embedding model, store the result in a persistent Chroma
collection, and rebuild the BM25 keyword index used for hybrid search.

Usage:
    python ingest.py

Safe to re-run any time you add, remove, or change PDFs in data/pdfs/ --
each chunk's id is derived from its content, so re-ingesting an unchanged
PDF just overwrites the same ids (upsert), and new PDFs are added
alongside existing ones. It does not currently delete chunks for PDFs
you've removed from data/pdfs/ -- see the README for how to clear the
index if you need that.
"""
import hashlib
import sys
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader

import config
from chunking import chunk_text
from hybrid_search import build_bm25_index


def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(name=config.COLLECTION_NAME)


def extract_pages(pdf_path: Path):
    """Yield (page_number, text) for each page that has extractable text.

    Pages with no extractable text (e.g. scanned images) are skipped --
    those need OCR first (see README) before this pipeline can use them.
    """
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield i, text


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def embed_batch(client: ollama.Client, texts):
    response = client.embed(model=config.EMBED_MODEL, input=texts)
    return response.embeddings


def generate_chunk_context(client: ollama.Client, doc_title: str, page_num: int, page_text: str, chunk: str) -> str:
    """Ask the chat model for a short sentence situating this chunk within
    its page, to prepend to the embedding input. Only called when
    CONTEXTUALIZE_CHUNKS=true -- see config.py for the cost tradeoff.
    """
    prompt = (
        f"Document: {doc_title}\n"
        f"Full text of page {page_num}:\n{page_text}\n\n"
        f"Excerpt from that page:\n{chunk}\n\n"
        "In one short sentence (max 25 words), describe what this excerpt "
        "covers and how it relates to the page, so someone could tell what "
        "it's about without seeing the rest of the page. Reply with only "
        "that sentence, nothing else."
    )
    try:
        response = client.chat(model=config.CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
        return response.message.content.strip()
    except Exception as e:
        print(f"    [!] chunk-context generation failed, continuing without it: {e}")
        return ""


def build_embed_text(doc_title: str, page_num: int, chunk: str, llm_context: str = "") -> str:
    """The text actually sent to the embedding model: clean chunk text
    prefixed with cheap deterministic context (always) and, optionally,
    an LLM-generated context sentence. The chunk stored in Chroma's
    `documents` field stays as the clean, unprefixed text -- this prefix
    only affects what gets embedded, to help the embedding capture which
    document/section a chunk belongs to without polluting what the
    answering model actually reads.
    """
    header = f"Document: {doc_title} | Page {page_num}"
    if llm_context:
        header += f"\nContext: {llm_context}"
    return f"{header}\n\n{chunk}"


def ingest_pdf(pdf_path: Path, collection, ollama_client: ollama.Client = None):
    """Extract, chunk, (optionally contextualize,) embed, and store one
    PDF. Returns chunk count."""
    ollama_client = ollama_client or ollama.Client(host=config.OLLAMA_HOST)
    doc_title = pdf_path.stem

    # records: list of (chunk_text, page_number, embed_text)
    records = []
    for page_num, page_text in extract_pages(pdf_path):
        for chunk in chunk_text(page_text, config.CHUNK_SIZE, config.OVERLAP_SENTENCES):
            llm_context = ""
            if config.CONTEXTUALIZE_CHUNKS:
                llm_context = generate_chunk_context(ollama_client, doc_title, page_num, page_text, chunk)
            embed_text = build_embed_text(doc_title, page_num, chunk, llm_context)
            records.append((chunk, page_num, embed_text))

    if not records:
        print(f"  [!] No extractable text in {pdf_path.name} -- "
              f"looks like a scanned/image PDF, needs OCR first.")
        return 0

    all_embeddings = []
    embed_texts = [r[2] for r in records]
    for i, batch in enumerate(batched(embed_texts, config.EMBED_BATCH_SIZE)):
        all_embeddings.extend(embed_batch(ollama_client, batch))
        done = min((i + 1) * config.EMBED_BATCH_SIZE, len(embed_texts))
        print(f"    embedded {done}/{len(embed_texts)} chunks", end="\r")
    print()

    ids, documents, metadatas = [], [], []
    for idx, (chunk, page_num, _embed_text) in enumerate(records):
        chunk_id = hashlib.sha256(
            f"{doc_title}-p{page_num}-{idx}-{chunk[:80]}".encode("utf-8")
        ).hexdigest()
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({"source": pdf_path.name, "page": page_num})

    collection.upsert(
        ids=ids,
        embeddings=all_embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(records)


def main():
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {config.PDF_DIR}")
        print("Drop .pdf files there and re-run: python ingest.py")
        return

    if config.CONTEXTUALIZE_CHUNKS:
        print("CONTEXTUALIZE_CHUNKS is on: this will make one extra chat-model "
              "call per chunk during ingestion, which is much slower for "
              "hundreds of pages. Turn it off in .env if this is too slow.\n")

    try:
        collection = get_chroma_collection()
    except Exception as e:
        print(f"Failed to open Chroma collection: {e}")
        sys.exit(1)

    ollama_client = ollama.Client(host=config.OLLAMA_HOST)
    total = 0
    for pdf_path in pdfs:
        print(f"Ingesting {pdf_path.name} ...")
        try:
            n = ingest_pdf(pdf_path, collection, ollama_client)
        except Exception as e:
            print(f"  [!] Failed to ingest {pdf_path.name}: {e}")
            print("      Is Ollama running (`ollama serve`) and is the "
                  f"embedding model pulled (`ollama pull {config.EMBED_MODEL}`)?")
            continue
        print(f"  -> {n} chunks stored")
        total += n

    print("\nRebuilding BM25 keyword index for hybrid search...")
    build_bm25_index(collection)

    print(f"\nDone. {total} chunks in collection '{config.COLLECTION_NAME}' "
          f"at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
