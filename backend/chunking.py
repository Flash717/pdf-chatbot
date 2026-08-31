"""
Paragraph/sentence-aware text chunking.

The previous version of this project split text into fixed-size character
windows, which regularly cut sentences (and even words) in half -- a chunk
ending mid-sentence embeds poorly, because the embedding has to represent a
fragment that doesn't stand on its own.

This version instead: (1) splits on paragraph breaks where the PDF
preserves them, (2) within each paragraph, splits into sentences, and
(3) greedily packs whole sentences into ~CHUNK_SIZE-character chunks,
never cutting a sentence, with a couple of sentences of overlap carried
into the next chunk so context isn't lost right at a chunk boundary.

Known limitation: the sentence splitter is a regex heuristic, not a real
NLP sentence tokenizer -- it will occasionally mis-split on abbreviations
like "Dr." or "e.g.". A library like nltk's punkt or spaCy would handle
this more precisely at the cost of a heavier dependency; this trade-off
seemed reasonable for keeping the project dependency-light, but swap it in
`split_into_sentences` below if your documents are abbreviation-heavy and
you're seeing bad splits there.
"""
import re

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])')
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse hard line-wraps and repeated whitespace into single spaces.

    pypdf's extract_text() inserts a newline at every visual line wrap in
    the PDF, not at sentence or paragraph boundaries -- so a single
    sentence is often split across several lines. This undoes that before
    sentence splitting runs.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_into_sentences(text: str):
    """Split text into a flat list of sentences, respecting paragraph
    breaks (a paragraph break is never merged across by the sentence
    splitter, even if the last sentence before it doesn't end in
    punctuation)."""
    text = text.strip()
    if not text:
        return []

    sentences = []
    for para in _PARAGRAPH_SPLIT_RE.split(text):
        para = _normalize(para)
        if not para:
            continue
        parts = _SENTENCE_SPLIT_RE.split(para)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def chunk_sentences(sentences, size: int, overlap_sentences: int = 2):
    """Greedily pack sentences into chunks of up to ~`size` characters.

    Never splits a sentence -- if a single sentence is longer than
    `size`, it becomes its own (oversized) chunk rather than being cut,
    since a truncated sentence is worse for retrieval than one long
    chunk. Consecutive chunks share the last `overlap_sentences`
    sentences of the previous chunk, so a fact stated right at a chunk
    boundary isn't orphaned from its context.
    """
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    def flush():
        if current:
            chunks.append(" ".join(current).strip())

    for sentence in sentences:
        if current and current_len + len(sentence) + 1 > size:
            flush()
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(s) + 1 for s in current)
        current.append(sentence)
        current_len += len(sentence) + 1

    flush()
    return chunks


def chunk_text(text: str, size: int, overlap_sentences: int = 2):
    """Split raw page text into paragraph/sentence-aware chunks."""
    sentences = split_into_sentences(text)
    return chunk_sentences(sentences, size, overlap_sentences)
