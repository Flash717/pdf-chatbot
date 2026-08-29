"""
Extract text from every PDF in PDF_DIR, split it into overlapping chunks,
embed each chunk with a local Ollama embedding model, and store the
result in a persistent Chroma collection on disk.

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


def chunk_text(text: str, size: int, overlap: int):
    """Split text into overlapping character-based chunks."""
    if size <= overlap:
        raise ValueError("CHUNK_SIZE must be greater than CHUNK_OVERLAP")
    chunks = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def embed_batch(client: ollama.Client, texts):
    response = client.embed(model=config.EMBED_MODEL, input=texts)
    return response.embeddings


def ingest_pdf(pdf_path: Path, collection, ollama_client: ollama.Client = None):
    """Extract, chunk, embed, and store one PDF. Returns chunk count."""
    ollama_client = ollama_client or ollama.Client(host=config.OLLAMA_HOST)
    doc_id = pdf_path.stem

    records = []  # list of (chunk_text, page_number)
    for page_num, page_text in extract_pages(pdf_path):
        for chunk in chunk_text(page_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            records.append((chunk, page_num))

    if not records:
        print(f"  [!] No extractable text in {pdf_path.name} -- "
              f"looks like a scanned/image PDF, needs OCR first.")
        return 0

    all_embeddings = []
    texts_only = [r[0] for r in records]
    for i, batch in enumerate(batched(texts_only, config.EMBED_BATCH_SIZE)):
        all_embeddings.extend(embed_batch(ollama_client, batch))
        done = min((i + 1) * config.EMBED_BATCH_SIZE, len(texts_only))
        print(f"    embedded {done}/{len(texts_only)} chunks", end="\r")
    print()

    ids, metadatas = [], []
    for idx, (chunk, page_num) in enumerate(records):
        chunk_id = hashlib.sha256(
            f"{doc_id}-p{page_num}-{idx}-{chunk[:80]}".encode("utf-8")
        ).hexdigest()
        ids.append(chunk_id)
        metadatas.append({"source": pdf_path.name, "page": page_num})

    collection.upsert(
        ids=ids,
        embeddings=all_embeddings,
        documents=texts_only,
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

    print(f"\nDone. {total} chunks in collection '{config.COLLECTION_NAME}' "
          f"at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
