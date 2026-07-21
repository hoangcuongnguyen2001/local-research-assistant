
# Local Research & CTI Triage Engine

A local, offline research-paper triage dashboard powered by a self-hosted LLM (via [Ollama](https://ollama.com)). Drop in papers (PDF, DOCX, HTML, or a URL) and it produces a grounded Markdown digest for each one, with a self-verification pass, structured metadata, and citation export. Every processed paper also gets its own retrieval-grounded chat, plus search and comparison across your whole library.

Everything runs locally - no data leaves the machine. Built with [FastAPI](https://fastapi.tiangolo.com/) and calls Ollama's native API directly (no agent framework).

## Features

- **Multi-format ingest**: PDF, TXT, MD, DOCX, HTML, or paste a URL (fetched server-side and staged automatically).
- **Digest pipeline**: map-reduce summarization (chunk → per-batch summary → combined structural digest) so long papers don't need to be stuffed into one giant prompt.
- **Self-verification**: after each digest is generated, a second pass checks its claims against the source material and appends a "Verification Notes" section flagging anything unsupported - advisory, not blocking.
- **Structured metadata**: title, authors, venue, venue type, year, DOI, and abstract are extracted for every paper and shown in the dashboard.
- **Per-paper chat (RAG)**: ask follow-up questions grounded in the actual paper text. Each paper's content is chunked and embedded once (cached), and each question retrieves only the most relevant chunks rather than stuffing the whole paper - faster and more focused. Conversations persist across restarts.
- **Library search**: search across every processed paper's content at once, ranked by semantic similarity, with source attribution per result.
- **Cross-paper comparison**: select 2+ papers and get a synthesized comparison of their methodology, findings, and conclusions.
- **Citation export**: generate a citation for any paper in APA, Harvard, ACL, or IEEE style, plus a BibTeX entry (also importable into Zotero). Formatting is deterministic Python, not a model call - no hallucination risk on something this mechanical.

## How it works

1. **Ingest** a document - upload a `.pdf` / `.txt` / `.md` / `.docx` / `.html` file, or paste a URL - into `staging/`.
2. **Trigger the pipeline** (button in the UI, or `POST /api/run`). For each staged file:
   - `runtime_processing/` is wiped clean first, then the file is moved there in isolation.
   - Text is extracted (PDF via `pypdf`, with a fallback between "layout" and "plain" extraction modes per page - layout mode avoids mid-word splitting on some PDFs but can fail badly on two-column layouts, so whichever mode returns substantially more text per page wins).
   - The digest is generated via **map-reduce**: the text is chunked, grouped into batches sized to stay under the model's GPU-resident context ceiling, each batch is summarized, and the partial summaries are combined into one structured digest.
   - A **verification pass** compares the final digest against the same partial summaries and appends any flagged unsupported claims.
   - **Metadata** (title/authors/venue/year/DOI/abstract) is extracted from the document's front matter.
   - The source file is archived to `library/` (not deleted), so it's available for chat/search/citation later.
3. **Browse, chat, search, compare, and cite** through the dashboard.

The app talks to a local Ollama server at `http://localhost:11434`, using its **native** `/api/chat` and `/api/embeddings` endpoints directly (not the OpenAI-compatible shim), because that's the only way to control `num_ctx` per request.

### A note on performance

On memory-constrained GPUs, the requested context window (`num_ctx`) determines how much of the model can stay GPU-resident - not just how much conversation fits. On this project's dev machine (an 8B Q4_K_M model, 6GB VRAM), the model ran ~72%/28% CPU/GPU at Ollama's default 131072-token context, but was 100% GPU-resident once `num_ctx` was capped at 7168 (empirically found via `ollama ps`; the model stays fully GPU-resident up to ~7168 tokens and falls back to heavy CPU offload above ~8192 on this hardware). Every generation call in this app - chat, digest batches, verification, metadata - deliberately stays under that ceiling. If you're running on different hardware, re-check this with `ollama ps` while generating and adjust `GENERATION_NUM_CTX` in `agent.py` accordingly.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally, with:
  - A generation model pulled (default `gemma4`; `ollama pull gemma4`) - swap `GENERATION_MODEL` in `agent.py` for whatever you have.
  - The embedding model: `ollama pull nomic-embed-text`
- Python packages:
  ```
  fastapi
  uvicorn
  pypdf
  requests
  numpy
  python-docx
  beautifulsoup4
  lxml
  ```

## Running it

```bash
pip install fastapi uvicorn pypdf requests numpy python-docx beautifulsoup4 lxml
ollama serve                 # if not already running
ollama pull gemma4
ollama pull nomic-embed-text
python -m uvicorn agent:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in a browser.

## Directory layout

| Path | Purpose |
| --- | --- |
| `staging/` | Drop zone for newly uploaded/fetched documents, awaiting a pipeline run. |
| `runtime_processing/` | Scratch space holding the single document currently being analyzed. Cleared before each file and after each run. |
| `result/` | One `<original-name>-summary.md` per processed document - the generated digest plus verification notes. |
| `library/` | Retained original source documents (post-processing), used for chat grounding, citation extraction, and library search. |
| `embeddings/` | One `<name>.json` chunk+embedding index per paper, built once and cached. |
| `chat_sessions/` | One `<name>.json` conversation history per paper, so chats survive a restart. |
| `metadata/` | One `<name>.json` of extracted bibliographic metadata per paper. |
| `agent.py` | The whole app: FastAPI endpoints, background pipeline, RAG/search/compare/citation logic, and the dashboard UI. |
| `offline_digest.md` | Example/reference digest output. |

## API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Serves the HTML dashboard. |
| `/api/upload` | POST | Uploads a file (multipart `file` field) into `staging/`. |
| `/api/upload-url` | POST | Fetches a URL server-side (`{"url": "..."}`) and stages it as PDF/HTML/TXT based on content type. |
| `/api/run` | POST | Kicks off the background pipeline over all staged files. Returns 400 if a run is already in progress. |
| `/api/status` | GET | Current pipeline status (`Idle` / `Processing` / `Complete`), current file, and a running log (includes live map-reduce batch progress). |
| `/api/results` | GET | Lists generated `*-summary.md` files in `result/`. |
| `/api/results/{filename}` | GET | Returns the Markdown content of a given summary file. |
| `/api/metadata` | GET | Bulk metadata for every paper, keyed by base filename. |
| `/api/citation/{base_name}` | GET | `?style=apa\|harvard\|acl\|ieee` - returns the formatted citation and a BibTeX entry. Lazily extracts metadata if missing. |
| `/api/library-search` | GET | `?q=...&k=8` - semantic search across every paper's cached chunks. |
| `/api/compare` | POST | `{"filenames": [...]}` (2+ base names) - synthesizes a comparison across their digests. |
| `/api/chat/{filename}/history` | GET | Returns the stored chat turns for a paper. |
| `/api/chat/{filename}` | POST | `{"message": "..."}` - asks a RAG-grounded question, returns the assistant's reply. |
| `/api/chat/{filename}` | DELETE | Clears a paper's chat history. |

`{filename}`/`{base_name}` above is the digest's base name with no extension (e.g. `2108.05080v4`), which is how papers are keyed consistently across `library/`, `embeddings/`, `chat_sessions/`, and `metadata/`.

## Notes / known limitations

- Only one pipeline run can be active at a time; `/api/run` rejects concurrent triggers.
- `runtime_processing/` is single-document isolation, not a queue - each file is fully processed and evicted before the next begins.
- No cache invalidation: if a source PDF is replaced under the same filename, its cached `embeddings/`/`chat_sessions/`/`metadata/` won't auto-refresh.
- No cross-document vector store beyond what `library-search` already does at query time (it re-embeds/caches lazily per paper rather than maintaining one merged index) - fine at small-library scale, would need revisiting at real scale.
- Single-user, local tool - no auth, no multi-user session isolation.
- The dashboard auto-refreshes status every 2s and the results list every 4s.
