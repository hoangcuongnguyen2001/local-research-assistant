
# Local Research & CTI Triage Engine

A local, offline document-triage dashboard that uses a self-hosted LLM (via [Ollama](https://ollama.com)) to summarize research papers, reports, or CTI documents dropped into a watch folder. Built with [FastAPI](https://fastapi.tiangolo.com/) and [smolagents](https://github.com/huggingface/smolagents), it runs an isolated agent per document and writes a Markdown digest for each one.

Everything runs locally - no data leaves the machine.

## How it works

1. **Upload** a `.pdf`, `.txt`, or `.md` file through the dashboard. It's saved to `staging/`.
2. **Trigger the pipeline** (button in the UI, or `POST /api/run`). For each staged file:
   - `runtime_processing/` is wiped clean of any leftover files first, so each run starts from a pristine slate.
   - The file is moved from `staging/` into `runtime_processing/` (isolating it as the single active document for that run).
   - A fresh `CodeAgent` instance is created (no shared state between documents) and given two tools:
     - `read_local_article` — reads the isolated file's contents (with PDF text extraction via `pypdf`).
     - `write_individual_digest` — appends the agent's summary to `result/<name>-summary.md`.
   - The agent is prompted to read the file and produce a dense technical summary.
   - After the agent finishes, its Python object is deleted and garbage-collected (to release any file locks Windows may be holding), and the file is removed from `runtime_processing/`.
3. **Browse results** in the dashboard's Output Folder panel, or via `GET /api/results`.

The agent talks to a local model server at `http://localhost:11434/v1` (Ollama's OpenAI-compatible API), using the `gemma4:latest` model by default.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the `gemma4` model pulled (`ollama pull gemma4`)
- Python packages:
  ```
  fastapi
  uvicorn
  pypdf
  smolagents
  ```

## Running it

```bash
pip install fastapi uvicorn pypdf smolagents
ollama serve                 # if not already running
ollama pull gemma4
python -m uvicorn agent:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in a browser.

## Directory layout

| Path | Purpose |
| --- | --- |
| `staging/` | Drop zone for newly uploaded documents, awaiting a pipeline run. |
| `runtime_processing/` | Scratch space holding the single document currently being analyzed. Cleared before each file and after each run. |
| `result/` | One `<original-name>-summary.md` per processed document, containing the agent's Markdown digest. |
| `agent.py` | FastAPI app: API endpoints, background processing pipeline, and the dashboard UI. |
| `offline_digest.md` | Example/reference digest output. |

## API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Serves the HTML dashboard. |
| `/api/upload` | POST | Uploads a file (multipart `file` field) into `staging/`. |
| `/api/run` | POST | Kicks off the background pipeline over all staged files. Returns 400 if a run is already in progress. |
| `/api/status` | GET | Current pipeline status (`Idle` / `Processing` / `Complete`), current file, and a running log. |
| `/api/results` | GET | Lists generated `*-summary.md` files in `result/`. |
| `/api/results/{filename}` | GET | Returns the Markdown content of a given summary file. |

## Notes

- Only one pipeline run can be active at a time; `/api/run` rejects concurrent triggers.
- `runtime_processing/` is treated as single-document isolation, not a queue — each file is fully processed and evicted before the next begins.
- The dashboard auto-refreshes status every 2s and the results list every 4s.

