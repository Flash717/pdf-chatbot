# PDF Chatbot

A retrieval-augmented chatbot that answers questions from your own PDFs,
running entirely on free, local models via [Ollama](https://ollama.com) --
no API keys, no per-request cost. Python (FastAPI) backend, a small
embeddable JS widget for the frontend.

## How it works

1. `backend/ingest.py` extracts text from every PDF in `data/pdfs/`, splits
   it into paragraph/sentence-aware chunks (`backend/chunking.py`), embeds
   each chunk (with a bit of prepended document/page context) using a
   local Ollama embedding model, stores the vectors in a persistent
   [Chroma](https://www.trychroma.com) database on disk (`chroma_db/`),
   and rebuilds a BM25 keyword index (`backend/hybrid_search.py`) used
   alongside vector search.
2. `backend/main.py` is a FastAPI server. `POST /chat` runs the retrieval
   pipeline described below and asks a local Ollama chat model to answer
   using only the retrieved text -- with source file + page citations.
3. `frontend/public/widget.js` is a plain, dependency-free JS file you
   embed on any web page with a `<script>` tag. It renders a chat bubble
   that calls your backend's `/chat` endpoint.

Adding more PDFs later is just: drop them in `data/pdfs/`, re-run
`ingest.py`. There's no hard limit on number of documents baked into this
design -- Chroma is built to hold a large number of chunks, though very
large libraries (many thousands of pages) may eventually want a beefier
vector database; see "Scaling further" below.

## Retrieval pipeline

Retrieval quality is what actually determines answer quality in a system
like this, so `backend/rag.py` does more than a single vector-similarity
lookup. Per question:

1. **Chunking (at ingest time).** `chunking.py` splits each page into
   sentences (undoing pypdf's mid-sentence line-wrap breaks first) and
   greedily packs whole sentences into ~`CHUNK_SIZE`-character chunks,
   never cutting a sentence, with a couple of sentences of overlap
   carried across chunk boundaries. This replaces blind fixed-size
   character windows, which regularly split sentences (and facts) in
   half.
2. **Contextual embedding (at ingest time).** Before embedding, each
   chunk is prefixed with `Document: <filename> | Page: <n>` so the
   embedding captures which document/page it's from, not just the raw
   sentence. Set `CONTEXTUALIZE_CHUNKS=true` to additionally have the
   chat model write a one-sentence summary of how each chunk fits its
   page and prepend that too -- meaningfully better for retrieval on
   large multi-document corpora, but one extra local LLM call per chunk,
   which is slow across hundreds of pages. The chunk text stored and
   later shown to the answering model stays clean/unprefixed either way.
3. **Hybrid search.** Vector search (Chroma) and BM25 keyword search
   (`hybrid_search.py`) each retrieve `RETRIEVE_CANDIDATES` chunks, then
   the two ranked lists are merged with Reciprocal Rank Fusion. Vector
   search is good at paraphrase/meaning but often misses exact terms --
   names, model numbers, specific dates -- that embeddings blur; BM25
   catches those. A chunk found by both searches naturally outranks one
   found by only one.
4. **Reranking.** If `RERANK_ENABLED=true` (default), the fused candidate
   pool is handed to the local chat model, which ranks them by actual
   relevance to the question using structured JSON output -- generally
   more accurate than either search's own ranking, at the cost of one
   more local LLM call per question. If that call fails or returns
   something unparseable, retrieval falls back to the fused order rather
   than erroring.
5. **Answering.** The final `TOP_K` chunks go into the prompt as context,
   with the model instructed to answer only from that context and cite
   source file + page.

Every stage is tunable via env vars in `.env` -- see `.env.example`. If
retrieval still feels weak on your documents after this, the next things
worth trying are a stronger embedding model, and building a small set of
test questions with known-correct source pages so you can actually
measure whether a change helped (see "What's verified" below for why I'm
flagging measurement specifically -- none of this was tunable against
your real documents from inside this sandbox).

## 1. Install and start Ollama

Download Ollama from https://ollama.com and start it (`ollama serve`, or
it may already run as a background service depending on your OS/install
method). Then pull the two models this project uses by default:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

I'm not fully certain these are still the best/most current small free
models in Ollama's library as you read this -- their catalog changes.
Check https://ollama.com/library and run `ollama list` to see what you
have; swap `EMBED_MODEL` / `CHAT_MODEL` in `.env` if you pick different
ones. `llama3.1` is a fairly large model -- if it's too slow on your
machine, look for a smaller instruct model in the library (e.g. a 3B/7B
class model) and swap it in the same way.

## 2. Set up and run the backend

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
cd backend
uv venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

cp .env.example .env            # optional, defaults work for local dev

# Put your PDFs in data/pdfs/ (from the project root), then:
uv run ingest.py

uv run uvicorn main:app --reload --port 8000
```

Visit http://localhost:8000/docs for interactive API docs. Try:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about X?"}'
```

### Adding more PDFs later

Drop new `.pdf` files into `data/pdfs/` and re-run `python ingest.py` --
it only needs to touch the new/changed files' worth of embeddings (it
re-embeds a whole PDF if you re-run it on an unchanged one, which is
wasteful but harmless). There's also a `POST /ingest` endpoint that
accepts file uploads directly, if you'd rather let an admin UI or script
add PDFs without shell access to the server.

Removing a PDF from `data/pdfs/` does **not** currently remove its chunks
from the index -- if you need that, delete `chroma_db/` and
`bm25_index.pkl` and re-run `ingest.py` against whatever PDFs remain.

## 3. Run/embed the frontend

For local testing:

```bash
cd frontend
npm install
npm start
```

Open http://localhost:3000/demo.html -- the chat bubble in the corner
talks to your backend.

To embed on your real website, copy `widget.js` and `widget.css` (or
serve them from wherever your site's static assets live) and add:

```html
<link rel="stylesheet" href="/path/to/widget.css">
<script src="/path/to/widget.js" data-api-url="https://your-backend-domain.example.com"></script>
```

`data-api-url` should point at wherever you've deployed the FastAPI
backend (see below). Before going live, also lock down CORS in
`backend/main.py` -- it currently allows all origins (`allow_origins=["*"]`),
which is fine for local dev but should be your real site's origin(s)
once this is public.

## Important: hosting/deployment caveat

Because this uses Ollama for free local inference, **the backend must run
on a machine that has Ollama installed and enough RAM (and ideally a GPU)
to run your chosen models** -- not a typical serverless platform (Vercel
functions, AWS Lambda, etc.), which won't have Ollama available. Practical
options:

- A VPS or dedicated server you control (e.g. a cloud VM) with Ollama
  installed alongside the FastAPI app.
- A container/VM with GPU access if you want faster responses.
- Keeping it running on your own machine/network if the website's
  audience is small/internal and you're comfortable exposing that
  backend (e.g. via a reverse proxy/tunnel).

If down the line you want simpler hosting (serverless, no GPU box to
manage) at the cost of no longer being free, swapping `OLLAMA_HOST`-based
calls in `backend/rag.py` / `backend/ingest.py` for a hosted API (Anthropic,
OpenAI, etc.) is a relatively contained change, since retrieval (Chroma)
is decoupled from generation.

## Scaling further

- Chroma here runs embedded/local (`PersistentClient`), which is simple
  and free but lives on one machine's disk. If you outgrow that (very
  large document counts, need for multiple backend instances sharing one
  index), look at Chroma's client/server mode or a managed vector DB
  (Pinecone, Qdrant Cloud, etc.) -- the `ingest.py`/`rag.py` split keeps
  that swap contained to `get_chroma_collection()`.
- If you turn on `CONTEXTUALIZE_CHUNKS`, ingesting hundreds of pages means
  thousands of extra local chat-model calls -- expect ingestion to take
  much longer, and consider a smaller/faster `CHAT_MODEL` for that step.
- Scanned/image-only PDFs won't yield any extractable text via `pypdf`
  and are skipped with a warning during ingestion -- they'd need OCR
  first (Claude's PDF tooling or a library like `pytesseract` can do
  this) before this pipeline can index them.

## What's verified vs. what you should check yourself

I built and tested this in a sandbox without Ollama available (it's a
model-serving daemon meant to run on your own machine), so here's exactly
what was and wasn't verified end-to-end:

- **Verified directly**: the Chroma/Ollama/`rank_bm25` Python library API
  calls used in this code (`PersistentClient`, `get_or_create_collection`,
  `upsert`, `query`/`get` including their `ids` fields, `ollama.Client()
  .chat()/.embed()` and their response field names, `ollama.chat`'s
  structured-JSON `format` parameter, and `BM25Okapi`) were checked
  against the actually-installed package versions and, for the Ollama
  client, its source code -- not guessed from memory. The new sentence-
  aware chunker was tested against realistic pypdf-style text (including
  mid-sentence line wraps and an abbreviation edge case, which is called
  out as a known limitation in `chunking.py`). Chroma storage/retrieval
  and BM25 index build/search were tested against a real generated PDF.
  The full FastAPI request/response flow (`/health`, `/chat`, validation)
  was tested with the Ollama calls mocked, confirming the wiring between
  hybrid retrieval, fusion, and response schema is correct. The Node/
  Express static server and widget files were confirmed to serve
  correctly.
- **Not verified here, please check on your machine**: an actual live
  call through Ollama (embedding + chat generation), so in particular:
  whether your chat model reliably follows the structured-JSON rerank
  format (the code falls back gracefully if not, but you should confirm
  reranking is actually engaging rather than silently always falling
  back), the real answer quality/latency you'll get from `nomic-embed-text`
  / `llama3.1` on your hardware, and how much the three retrieval
  improvements actually help on your specific documents. Also
  double-check that those two model names are still current in Ollama's
  library when you set this up -- I flagged this above because model
  catalogs change over time and I can't confirm what's current beyond my
  own knowledge cutoff.

## Project layout

```
pdf-chatbot/
├── backend/
│   ├── main.py           FastAPI app (/health, /chat, /ingest)
│   ├── rag.py             hybrid retrieval + fusion + rerank + answer generation
│   ├── ingest.py          PDF -> chunks -> embeddings -> Chroma + BM25
│   ├── chunking.py        paragraph/sentence-aware chunking
│   ├── hybrid_search.py   BM25 index + Reciprocal Rank Fusion
│   ├── config.py          env-var driven configuration
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── server.js          local dev static server (optional)
│   ├── package.json
│   └── public/
│       ├── widget.js       the embeddable chat widget
│       ├── widget.css
│       └── demo.html       example page embedding the widget
├── data/pdfs/             put your PDFs here
├── chroma_db/              persisted vector index (created on first ingest)
└── bm25_index.pkl          persisted BM25 keyword index (created on first ingest)
```
