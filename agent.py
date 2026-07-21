import os
import re
import shutil
import json
import requests
import numpy as np
import docx
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse
import pypdf

app = FastAPI(title="Local Agent Digest Dashboard")

GENERATION_MODEL = "gemma4:latest"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
# This model's weights (Q4_K_M, ~9.6GB) don't fit in this hardware's 6GB VRAM at Ollama's
# default 131072-token context reservation, so it runs mostly on CPU (measured ~72%/28%
# CPU/GPU) - 4-8x slower than full GPU residency. Capping num_ctx is the lever: measured
# via `ollama ps` that this model stays 100% GPU-resident up to ~7168 tokens of context,
# and falls back to heavy CPU offload above ~8192. Every direct call below stays under
# that ceiling on purpose.
GENERATION_NUM_CTX = 7168

def ollama_chat(messages: list[dict], num_ctx: int = GENERATION_NUM_CTX) -> str:
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": GENERATION_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": num_ctx},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_EMBEDDINGS_URL = "http://localhost:11434/api/embeddings"
CHUNK_WORDS = 200
CHUNK_OVERLAP_WORDS = 40
RETRIEVAL_TOP_K = 5
# Batches of ~8 chunks (~14K chars, ~3.5K tokens) verified to stay well under the
# GPU-resident ceiling above, with room left for the output tokens too.
MAP_REDUCE_BATCH_CHARS = 14000

PROCESSING_STATUS = {"status": "Idle", "current_file": None, "logs": []}

# Enforce clean workspace boundaries on your D: drive
os.makedirs("./staging", exist_ok=True)
os.makedirs("./runtime_processing", exist_ok=True)
os.makedirs("./result", exist_ok=True)
os.makedirs("./library", exist_ok=True)
os.makedirs("./chat_sessions", exist_ok=True)
os.makedirs("./embeddings", exist_ok=True)
os.makedirs("./metadata", exist_ok=True)

CHAT_SESSIONS: dict[str, list[dict]] = {}  # key: filename -> [{"role": "user"/"assistant", "content": str}, ...]
# In-memory cache, backed by a JSON file per session in ./chat_sessions so history
# survives a server restart. CHAT_SESSIONS is refilled from disk lazily, on first access.

def chat_session_path(filename: str) -> str:
    return os.path.join("./chat_sessions", f"{filename}.json")

def load_chat_session(filename: str) -> list[dict]:
    path = chat_session_path(filename)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_chat_session(filename: str, history: list[dict]) -> None:
    with open(chat_session_path(filename), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_chat_session(filename: str) -> list[dict]:
    if filename not in CHAT_SESSIONS:
        CHAT_SESSIONS[filename] = load_chat_session(filename)
    return CHAT_SESSIONS[filename]

# --- RAG: chunking + embedding retrieval, so chat only stuffs relevant excerpts, not the full paper ---
def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words])
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks

def embed_text(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_EMBEDDINGS_URL,
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]

def chunk_index_path(filename: str) -> str:
    return os.path.join("./embeddings", f"{filename}.json")

def get_or_build_chunk_index(filename: str, paper_text: str) -> list[dict]:
    """One chunk+embedding index per paper, cached to disk so it's computed only once."""
    path = chunk_index_path(filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    index = [{"text": chunk, "embedding": embed_text(chunk)} for chunk in chunk_text(paper_text)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f)
    return index

def top_k_chunks(question: str, index: list[dict], k: int = RETRIEVAL_TOP_K) -> list[str]:
    if not index:
        return []
    question_vec = np.array(embed_text(question))
    scored = []
    for entry in index:
        chunk_vec = np.array(entry["embedding"])
        similarity = float(np.dot(question_vec, chunk_vec) / (np.linalg.norm(question_vec) * np.linalg.norm(chunk_vec)))
        scored.append((similarity, entry["text"]))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [text for _, text in scored[:k]]

def read_file_content(active_dir: str, file_name: str) -> str:
    """Plain-function file reader shared by the pipeline tool and the chat endpoint."""
    file_path = os.path.join(active_dir, file_name)

    if not os.path.exists(file_path):
        return f"Error: File '{file_name}' not found in active directory '{active_dir}'."

    try:
        if file_name.lower().endswith('.pdf'):
            text_content = []
            # Use 'with' context manager so the binary file stream is closed instantly
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    # "layout" mode avoids the plain extractor's habit of inserting a
                    # spurious space mid-word at glyph-run boundaries (e.g. "FakeAVCeleb"
                    # -> "FakeA VCeleb"), so prefer it by default. But it can also
                    # catastrophically fail on multi-column PDFs (observed: ~5 chars/page
                    # on a 2-column ACL paper, vs. thousands with plain mode - a ~0.001
                    # length ratio), leaving the model with no real content to ground on.
                    # Normal page-to-page variation between the two modes stays within
                    # roughly 0.8-1.4x (observed), so only fall back to plain mode when
                    # layout comes back an order of magnitude shorter - a real failure,
                    # not just normal whitespace-handling differences.
                    plain = page.extract_text()
                    layout = page.extract_text(extraction_mode="layout")
                    text = plain if plain and len(layout) < 0.3 * len(plain) else layout
                    if text:
                        text_content.append(text)
            return "\n".join(text_content) if text_content else "PDF appeared empty."

        if file_name.lower().endswith('.docx'):
            document = docx.Document(file_path)
            paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
            return "\n".join(paragraphs) if paragraphs else "DOCX appeared empty."

        if file_name.lower().endswith(('.html', '.htm')):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return text if text else "HTML appeared empty."

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_name}: {str(e)}"

def write_digest_file(content: str, source_filename: str, summary_type: str) -> str:
    base_name = os.path.splitext(source_filename)[0]
    output_filename = f"{base_name}-summary.md"
    result_path = os.path.join("./result", output_filename)

    with open(result_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n## [{summary_type.upper()}] (Parsed from {source_filename})\n{content}\n")
    return result_path

def batch_chunks(chunks: list[str], max_chars: int = MAP_REDUCE_BATCH_CHARS) -> list[str]:
    """Group small chunks into larger batches, each still small enough to stay
    GPU-resident (see MAP_REDUCE_BATCH_CHARS), to cut down the number of model calls."""
    batches = []
    current: list[str] = []
    current_len = 0
    for chunk in chunks:
        if current and current_len + len(chunk) > max_chars:
            batches.append("\n\n---\n\n".join(current))
            current, current_len = [], 0
        current.append(chunk)
        current_len += len(chunk)
    if current:
        batches.append("\n\n---\n\n".join(current))
    return batches

def summarize_batch(file_name: str, batch_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an offline research assistant. Summarize the key technical points of this "
                "excerpt from a research paper in a dense paragraph. Ground strictly in the text - "
                "do not invent content."
            ),
        },
        {"role": "user", "content": f"--- EXCERPT FROM {file_name} ---\n{batch_text}\n--- END EXCERPT ---"},
    ]
    return ollama_chat(messages)

def combine_partial_summaries(file_name: str, partial_summaries: list[str]) -> str:
    combined = "\n\n---\n\n".join(partial_summaries)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an offline research assistant. Below are partial summaries covering "
                "different sections of the same paper, in order. Combine them into one dense, "
                "structural Markdown summary covering: core research area, architectural "
                "configurations or methodology, key findings, and limitations/future work. Ground "
                "every claim strictly in the partial summaries provided - do not invent content."
            ),
        },
        {
            "role": "user",
            "content": f"--- PARTIAL SUMMARIES FROM {file_name} ---\n{combined}\n--- END PARTIAL SUMMARIES ---",
        },
    ]
    return ollama_chat(messages)

def verify_digest(file_name: str, digest: str, partial_summaries: list[str]) -> str:
    """Lightweight fact-check: ask the model to compare the final digest against the same
    partial summaries it was built from, and flag anything unsupported. Advisory only - the
    verifier isn't infallible, so results get appended as notes rather than blocking the
    digest from being saved."""
    source = "\n\n---\n\n".join(partial_summaries)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a fact-checking assistant. Compare the FINAL SUMMARY against the SOURCE "
                "MATERIAL it was supposedly built from. List any specific claims, names, numbers, "
                "or findings in the final summary that are NOT supported by the source material. "
                "Be concise - a short bullet list, or the single line 'No unsupported claims "
                "detected.' if everything checks out. Do not repeat supported content."
            ),
        },
        {
            "role": "user",
            "content": (
                f"--- SOURCE MATERIAL (partial summaries of {file_name}) ---\n{source}\n"
                f"--- END SOURCE ---\n\n--- FINAL SUMMARY TO CHECK ---\n{digest}\n--- END FINAL SUMMARY ---"
            ),
        },
    ]
    return ollama_chat(messages)

def generate_digest(file_name: str, paper_text: str, on_progress=None) -> str:
    """Map-reduce summarization: batch the paper's chunks (small enough to stay
    GPU-resident per-call), summarize each batch, then combine the partial summaries into
    the final structured digest. Replaces a single giant stuffed-context call, which
    forced the model mostly onto CPU (much slower) and, via the old CodeAgent-based
    approach's truncation/drift, was prone to hallucinating unrelated content."""
    chunks = chunk_text(paper_text)
    if not chunks:
        return "No extractable text was found in this document."

    batches = batch_chunks(chunks)
    partial_summaries = []
    for i, batch in enumerate(batches, start=1):
        partial_summaries.append(summarize_batch(file_name, batch))
        if on_progress:
            on_progress(f"Summarized batch {i}/{len(batches)} for {file_name}.")

    if on_progress:
        on_progress(f"Combining {len(partial_summaries)} partial summaries for {file_name}.")
    final_digest = combine_partial_summaries(file_name, partial_summaries)

    if on_progress:
        on_progress(f"Verifying digest for {file_name}.")
    verification = verify_digest(file_name, final_digest, partial_summaries)

    return f"{final_digest}\n\n### Verification Notes\n{verification}"

def parse_json_loose(text: str) -> dict:
    """Models don't always follow a 'JSON only' instruction exactly (markdown fences, stray
    prose). Try a direct parse first, then fall back to pulling the first {...} block out."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}

def extract_metadata(file_name: str, paper_text: str) -> dict:
    """Title/authors/venue/year/abstract are almost always on the first page, so this only
    needs the front matter, not the whole paper - cheap enough to backfill existing papers too.
    venue_type/doi feed citation formatting (BibTeX entry type, etc.) downstream."""
    front_matter = paper_text[:3000]
    messages = [
        {
            "role": "system",
            "content": (
                "You extract bibliographic metadata from the start of a research paper. Respond "
                "with ONLY a JSON object (no markdown fences, no extra text) with exactly these "
                'keys: "title" (string), "authors" (array of strings, full names), "venue" '
                '(string, e.g. conference/journal name, empty string if unclear), "venue_type" '
                '(one of "conference", "journal", "preprint", "other" - your best guess from '
                'context, e.g. "arXiv" implies preprint), "year" (string, empty string if '
                'unclear), "doi" (string, empty string if not present), "abstract" (string, the '
                "paper's abstract if present, empty string otherwise). If a field can't be "
                "determined, use an empty string or empty array - never invent values."
            ),
        },
        {"role": "user", "content": f"--- START OF {file_name} ---\n{front_matter}\n--- END EXCERPT ---"},
    ]
    metadata = parse_json_loose(ollama_chat(messages))
    metadata.setdefault("title", "")
    metadata.setdefault("authors", [])
    metadata.setdefault("venue", "")
    metadata.setdefault("venue_type", "other")
    metadata.setdefault("year", "")
    metadata.setdefault("doi", "")
    metadata.setdefault("abstract", "")
    return metadata

def metadata_path(base_name: str) -> str:
    return os.path.join("./metadata", f"{base_name}.json")

def save_metadata(base_name: str, metadata: dict) -> None:
    with open(metadata_path(base_name), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_metadata(base_name: str) -> dict | None:
    path = metadata_path(base_name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- CITATION FORMATTING ---
# Deterministic (no model call): citation style rules are well-defined string formatting,
# not something that benefits from an LLM - doing it in plain code means zero hallucination
# risk and no latency, unlike everything else that touches the model in this file.

def _last_initials(full_name: str) -> str:
    """'Zhuohan Xie' -> 'Xie, Z.'"""
    parts = full_name.strip().split()
    if not parts:
        return full_name
    last = parts[-1]
    initials = " ".join(f"{p[0]}." for p in parts[:-1] if p)
    return f"{last}, {initials}".strip().rstrip(",")

def _initials_last(full_name: str) -> str:
    """'Zhuohan Xie' -> 'Z. Xie' (IEEE style)"""
    parts = full_name.strip().split()
    if not parts:
        return full_name
    last = parts[-1]
    initials = " ".join(f"{p[0]}." for p in parts[:-1] if p)
    return f"{initials} {last}".strip()

def format_citation_apa(metadata: dict) -> str:
    names = [_last_initials(a) for a in metadata.get("authors") or []]
    if not names:
        author_str = ""
    elif len(names) == 1:
        author_str = names[0]
    elif len(names) <= 20:
        author_str = ", ".join(names[:-1]) + ", & " + names[-1]
    else:
        author_str = ", ".join(names[:19]) + ", ... " + names[-1]
    year = metadata.get("year") or "n.d."
    title = metadata.get("title") or ""
    venue = metadata.get("venue") or ""
    return " ".join(p for p in [author_str, f"({year}).", f"{title}.", f"{venue}." if venue else ""] if p)

def format_citation_harvard(metadata: dict) -> str:
    names = [_last_initials(a) for a in metadata.get("authors") or []]
    author_str = ", ".join(names)
    year = metadata.get("year") or "n.d."
    title = metadata.get("title") or ""
    venue = metadata.get("venue") or ""
    return " ".join(p for p in [author_str, f"{year}.", f"{title}.", f"{venue}." if venue else ""] if p)

def format_citation_acl(metadata: dict) -> str:
    """ACL Anthology style uses full names and 'In <venue>.', not initials."""
    author_str = ", ".join(metadata.get("authors") or [])
    year = metadata.get("year") or "n.d."
    title = metadata.get("title") or ""
    venue = metadata.get("venue") or ""
    venue_part = f"In {venue}." if venue else ""
    return " ".join(p for p in [f"{author_str}." if author_str else "", f"{year}.", f"{title}.", venue_part] if p)

def format_citation_ieee(metadata: dict) -> str:
    names = [_initials_last(a) for a in metadata.get("authors") or []]
    author_str = ", ".join(names)
    title = metadata.get("title") or ""
    venue = metadata.get("venue") or ""
    year = metadata.get("year") or "n.d."
    venue_part = f"in {venue}, " if venue else ""
    return f'{author_str}, "{title}," {venue_part}{year}.'

CITATION_FORMATTERS = {
    "apa": format_citation_apa,
    "harvard": format_citation_harvard,
    "acl": format_citation_acl,
    "ieee": format_citation_ieee,
}

def format_citation(metadata: dict, style: str) -> str:
    formatter = CITATION_FORMATTERS.get(style.lower())
    if formatter is None:
        raise ValueError(f"Unknown citation style '{style}'. Choose from: {list(CITATION_FORMATTERS)}")
    return formatter(metadata)

def bibtex_key(base_name: str, metadata: dict) -> str:
    authors = metadata.get("authors") or []
    first_author_last = authors[0].split()[-1] if authors and authors[0].strip() else base_name
    year = metadata.get("year") or "nd"
    title_word = "".join(c for c in (metadata.get("title", "").split() or [""])[0] if c.isalnum())
    key = f"{first_author_last}{year}{title_word}"
    return re.sub(r'[^a-zA-Z0-9]', '', key) or base_name

def generate_bibtex(base_name: str, metadata: dict) -> str:
    venue_type = metadata.get("venue_type", "other")
    entry_type = {"conference": "inproceedings", "journal": "article", "preprint": "misc"}.get(venue_type, "misc")
    key = bibtex_key(base_name, metadata)
    authors_bib = " and ".join(metadata.get("authors") or []) or "Unknown"

    lines = [f"@{entry_type}{{{key},"]
    lines.append(f'  title = {{{metadata.get("title") or base_name}}},')
    lines.append(f'  author = {{{authors_bib}}},')
    if metadata.get("year"):
        lines.append(f'  year = {{{metadata["year"]}}},')
    if metadata.get("venue"):
        venue_field = "booktitle" if entry_type == "inproceedings" else "journal"
        lines.append(f'  {venue_field} = {{{metadata["venue"]}}},')
    if metadata.get("doi"):
        lines.append(f'  doi = {{{metadata["doi"]}}},')
    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)

# --- LOCK-PROOF BACKGROUND PROCESSING PIPELINE ---
def background_agent_pipeline():
    global PROCESSING_STATUS
    staging_dir = "./staging"
    run_dir = "./runtime_processing"
    
    valid_extensions = (".txt", ".md", ".pdf", ".docx", ".html", ".htm")
    staged_files = [f for f in os.listdir(staging_dir) if f.lower().endswith(valid_extensions)]
    
    if not staged_files:
        PROCESSING_STATUS["status"] = "Idle"
        PROCESSING_STATUS["logs"].append("Pipeline pass skipped: No staging items ready.")
        return

    PROCESSING_STATUS["status"] = "Processing"
    
    for file in staged_files:
        staging_path = os.path.join(staging_dir, file)
        runtime_path = os.path.join(run_dir, file)
        
        # 1. Clear any zombie files out of runtime first to ensure a pristine slate
        for old_zombie in os.listdir(run_dir):
            try:
                os.remove(os.path.join(run_dir, old_zombie))
            except Exception:
                pass # Continue if Windows is temporarily holding onto a link
                
        # 2. Move the target file into isolation
        try:
            shutil.move(staging_path, runtime_path)
            PROCESSING_STATUS["logs"].append(f"Isolated asset successfully: {file}")
        except Exception as move_err:
            PROCESSING_STATUS["logs"].append(f"Isolation shift failure for {file}: {str(move_err)}")
            continue

        PROCESSING_STATUS["current_file"] = file
        PROCESSING_STATUS["logs"].append(f"Starting fresh analysis layer for: {file}")

        try:
            paper_text = read_file_content(run_dir, file)
            digest_content = generate_digest(
                file, paper_text,
                on_progress=lambda msg: PROCESSING_STATUS["logs"].append(msg),
            )
            write_digest_file(digest_content, file, "structural summary matrix")
            PROCESSING_STATUS["logs"].append(f"Completed analysis matrix for {file}.")

            try:
                metadata = extract_metadata(file, paper_text)
                save_metadata(os.path.splitext(file)[0], metadata)
                PROCESSING_STATUS["logs"].append(f"Extracted metadata for {file}.")
            except Exception as meta_err:
                PROCESSING_STATUS["logs"].append(f"Metadata extraction warning for {file}: {str(meta_err)}")
        except Exception as e:
            PROCESSING_STATUS["logs"].append(f"Agent engine processing exception {file}: {str(e)}")
        finally:
            # Retain the source document in the library instead of deleting it,
            # so chat grounding has more to work with than the generated digest.
            if os.path.exists(runtime_path):
                try:
                    library_path = os.path.join("./library", file)
                    shutil.move(runtime_path, library_path)
                    PROCESSING_STATUS["logs"].append(f"Archived source asset to library: {file}")
                except Exception as move_err:
                    PROCESSING_STATUS["logs"].append(f"Post-run archive warning for {file}: {str(move_err)}")
            
    PROCESSING_STATUS["status"] = "Complete"
    PROCESSING_STATUS["current_file"] = None
    PROCESSING_STATUS["logs"].append("Pipeline run successfully completed.")

# --- API ENDPOINTS ---
@app.get("/api/status")
def get_status():
    return PROCESSING_STATUS

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    staging_path = os.path.join("./staging", file.filename)
    try:
        with open(staging_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        PROCESSING_STATUS["logs"].append(f"Staged file receipt validated: {file.filename}")
        return {"filename": file.filename, "status": "Successfully Staged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failure: {str(e)}")

MAX_URL_FETCH_BYTES = 50 * 1024 * 1024  # sanity cap, not a hard security boundary

@app.post("/api/upload-url")
def upload_from_url(payload: dict):
    url = payload.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")

    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > MAX_URL_FETCH_BYTES:
        raise HTTPException(status_code=400, detail="Remote file too large (>50MB).")

    content_type = response.headers.get("content-type", "").lower()
    parsed = urlparse(url)
    base_name = os.path.splitext(os.path.basename(parsed.path))[0] or parsed.netloc.replace(".", "-")
    base_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name)[:100] or "webpage"

    try:
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            filename = f"{base_name}.pdf"
            with open(os.path.join("./staging", filename), "wb") as f:
                f.write(response.content)
        elif "html" in content_type:
            filename = f"{base_name}.html"
            with open(os.path.join("./staging", filename), "w", encoding="utf-8") as f:
                f.write(response.text)
        else:
            filename = f"{base_name}.txt"
            with open(os.path.join("./staging", filename), "w", encoding="utf-8") as f:
                f.write(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stage fetched content: {str(e)}")

    PROCESSING_STATUS["logs"].append(f"Staged from URL: {url} -> {filename}")
    return {"filename": filename, "status": "Successfully Staged"}

@app.post("/api/run")
def trigger_pipeline(background_tasks: BackgroundTasks):
    global PROCESSING_STATUS
    if PROCESSING_STATUS["status"] == "Processing":
        raise HTTPException(status_code=400, detail="The agent loop is currently working.")
    PROCESSING_STATUS["logs"] = ["Initializing Agent Threads..."]
    background_tasks.add_task(background_agent_pipeline)
    return {"message": "Agent triggered."}

@app.get("/api/results")
def list_results():
    result_dir = "./result"
    if not os.path.exists(result_dir):
        return {"files": []}
    files = [f for f in os.listdir(result_dir) if f.endswith(".md")]
    return {"files": files}

@app.get("/api/results/{filename}")
def read_result_content(filename: str):
    safe_path = os.path.join("./result", filename)
    if not os.path.exists(safe_path) or not filename.endswith(".md"):
        raise HTTPException(status_code=404, detail="Summary target not located.")
    with open(safe_path, "r", encoding="utf-8") as f:
        return {"content": f.read()}

# --- METADATA ---
@app.get("/api/metadata")
def list_all_metadata():
    """Bulk fetch, keyed by base filename (no extension) - matches the digest/chat/embeddings
    naming convention so the frontend can look papers up with one key everywhere."""
    if not os.path.isdir("./metadata"):
        return {}
    result = {}
    for f in os.listdir("./metadata"):
        if f.endswith(".json"):
            base_name = os.path.splitext(f)[0]
            loaded = load_metadata(base_name)
            if loaded is not None:
                result[base_name] = loaded
    return result

@app.get("/api/citation/{base_name}")
def get_citation(base_name: str, style: str = "apa"):
    if style.lower() not in CITATION_FORMATTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown style '{style}'. Choose from: {list(CITATION_FORMATTERS)}",
        )

    metadata = load_metadata(base_name)
    if metadata is None:
        # Lazily extract - covers papers processed before this feature existed.
        library_filename = resolve_library_file(base_name)
        if library_filename is None:
            raise HTTPException(status_code=404, detail=f"No source document found for '{base_name}'.")
        paper_text = read_file_content("./library", library_filename)
        try:
            metadata = extract_metadata(library_filename, paper_text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Metadata extraction failed: {str(e)}")
        save_metadata(base_name, metadata)

    return {
        "formatted": format_citation(metadata, style),
        "bibtex": generate_bibtex(base_name, metadata),
        "style": style.lower(),
        "metadata": metadata,
    }

# --- LIBRARY SEARCH (corpus-wide, across every paper in ./library) ---
@app.get("/api/library-search")
def library_search(q: str, k: int = 8):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not os.path.isdir("./library"):
        return {"results": []}

    all_chunks = []  # (base_name, chunk_text, embedding)
    for f in os.listdir("./library"):
        base_name = os.path.splitext(f)[0]
        try:
            paper_text = read_file_content("./library", f)
            index = get_or_build_chunk_index(base_name, paper_text)
        except Exception:
            continue
        for entry in index:
            all_chunks.append((base_name, entry["text"], entry["embedding"]))

    if not all_chunks:
        return {"results": []}

    try:
        query_vec = np.array(embed_text(q))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query embedding failed: {str(e)}")

    scored = []
    for base_name, text, embedding in all_chunks:
        vec = np.array(embedding)
        similarity = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec)))
        scored.append((similarity, base_name, text))
    scored.sort(key=lambda item: item[0], reverse=True)

    return {
        "results": [
            {"source": base_name, "excerpt": text, "score": round(score, 4)}
            for score, base_name, text in scored[:k]
        ]
    }

# --- CROSS-PAPER COMPARISON ---
@app.post("/api/compare")
def compare_papers(payload: dict):
    filenames = payload.get("filenames", [])
    if not isinstance(filenames, list) or len(filenames) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 filenames to compare.")

    digests = []
    for base_name in filenames:
        result_path = os.path.join("./result", f"{base_name}-summary.md")
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail=f"No digest found for '{base_name}'.")
        with open(result_path, "r", encoding="utf-8") as f:
            digests.append((base_name, f.read()))

    combined = "\n\n===\n\n".join(f"--- DIGEST: {name} ---\n{content}" for name, content in digests)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an offline research assistant. Below are digests of multiple research "
                "papers. Synthesize a comparison: what they have in common, how they differ "
                "(methodology, findings, scope), and any notable complementary or conflicting "
                "conclusions. Ground every claim strictly in the digests provided - do not invent "
                "content."
            ),
        },
        {"role": "user", "content": combined},
    ]

    try:
        comparison = ollama_chat(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison generation failed: {str(e)}")

    return {"comparison": comparison, "papers": filenames}

# --- CHAT ENDPOINTS ---
def resolve_library_file(base_name: str) -> str | None:
    """Map a digest base name (no extension) back to its source file in ./library,
    which may carry any of the valid_extensions."""
    if not os.path.isdir("./library"):
        return None
    for f in os.listdir("./library"):
        if os.path.splitext(f)[0] == base_name:
            return f
    return None

@app.get("/api/chat/{filename}/history")
def get_chat_history(filename: str):
    return {"messages": get_chat_session(filename)}

@app.delete("/api/chat/{filename}")
def reset_chat_session(filename: str):
    CHAT_SESSIONS.pop(filename, None)
    path = chat_session_path(filename)
    if os.path.exists(path):
        os.remove(path)
    return {"message": f"Chat session cleared for {filename}."}

@app.post("/api/chat/{filename}")
def send_chat_message(filename: str, payload: dict):
    user_message = payload.get("message", "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message body cannot be empty.")

    library_filename = resolve_library_file(filename)
    if library_filename is None:
        raise HTTPException(status_code=404, detail=f"Source document for '{filename}' not found in library.")

    paper_text = read_file_content("./library", library_filename)

    try:
        chunk_index = get_or_build_chunk_index(filename, paper_text)
        relevant_chunks = top_k_chunks(user_message, chunk_index)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    history = get_chat_session(filename)
    history.append({"role": "user", "content": user_message})

    excerpts = "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else paper_text
    system_prompt = (
        "You are an offline research assistant answering questions about a specific paper. "
        "Below are the excerpts retrieved as most relevant to the current question. Ground your "
        "answer strictly in them. If the answer isn't in the excerpts, say so.\n\n"
        f"--- RELEVANT EXCERPTS FROM: {library_filename} ---\n{excerpts}\n--- END EXCERPTS ---"
    )
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": turn["role"], "content": turn["content"]} for turn in history
    ]

    try:
        assistant_content = ollama_chat(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat model call failed: {str(e)}")

    history.append({"role": "assistant", "content": assistant_content})
    save_chat_session(filename, history)
    return {"role": "assistant", "content": assistant_content}

# --- HTML VISUAL INTERFACE VIEW ---
@app.get("/", response_class=HTMLResponse)
def serve_dashboard_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Local Gemma Agent Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-gray-100 min-h-screen font-sans">
        <div class="max-w-7xl mx-auto p-6">
            <header class="flex flex-col md:flex-row md:justify-between md:items-center pb-6 border-b border-gray-800 gap-4">
                <div>
                    <h1 class="text-2xl font-bold tracking-tight text-white">Local Research & CTI Triage Engine</h1>
                    <p class="text-sm text-gray-400">Isolated Local Document Summary Extraction powered by Gemma 4 Edge</p>
                </div>
                <div class="flex items-center gap-4">
                    <span id="status-badge" class="px-3 py-1 rounded-full text-xs font-semibold bg-gray-800 text-gray-400">Idle</span>
                    <button onclick="toggleSearchPanel()" class="bg-gray-800 hover:bg-gray-700 text-gray-200 font-medium px-4 py-2 rounded-lg text-sm transition border border-gray-700">
                        Search Library
                    </button>
                    <button id="run-btn" onclick="runPipeline()" class="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 text-white font-medium px-4 py-2 rounded-lg text-sm transition">
                        Execute Agent Loop
                    </button>
                </div>
            </header>

            <main class="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
                <div class="space-y-6">
                    <div class="bg-gray-800/40 p-5 rounded-xl border border-gray-800">
                        <h2 class="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Ingest Document Drop-Zone</h2>
                        <div class="border-2 border-dashed border-gray-700 hover:border-indigo-500 rounded-lg p-6 text-center transition cursor-pointer relative">
                            <input type="file" id="file-uploader" accept=".pdf,.txt,.md,.docx,.html,.htm" onchange="uploadFile()" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
                            <svg class="mx-auto h-12 w-12 text-gray-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
                            <p class="text-xs text-gray-400">Select or drop PDF, TXT, MD, DOCX, or HTML documents here</p>
                            <p id="upload-feedback" class="text-xs text-indigo-400 mt-2 font-medium"></p>
                        </div>
                        <div class="flex gap-2 mt-3">
                            <input id="url-input" type="text" placeholder="Or paste a URL (PDF/HTML page)..." onkeydown="if(event.key==='Enter') uploadFromUrl();" class="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-indigo-500" />
                            <button id="url-upload-btn" onclick="uploadFromUrl()" class="bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 text-gray-200 font-medium px-3 py-2 rounded-lg text-xs transition">
                                Fetch
                            </button>
                        </div>
                    </div>

                    <div class="bg-gray-800/40 p-5 rounded-xl border border-gray-800 flex flex-col h-[345px]">
                        <div class="flex justify-between items-center mb-3">
                            <h2 class="text-sm font-semibold text-gray-300 uppercase tracking-wider">Output Folder (./result)</h2>
                            <button onclick="refreshResultsIndex()" class="text-xs text-indigo-400 hover:underline">Refresh</button>
                        </div>
                        <div id="results-list" class="flex-1 overflow-y-auto space-y-2 text-sm text-gray-400">
                            No compiled reports detected yet.
                        </div>
                        <button id="compare-btn" onclick="openComparePanel()" disabled class="hidden mt-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 text-white font-medium px-3 py-2 rounded-lg text-xs transition">
                            Compare Selected (0)
                        </button>
                    </div>
                </div>

                <div class="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-gray-800/40 p-5 rounded-xl border border-gray-800 flex flex-col h-[520px]">
                        <h2 class="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">Runtime Processing Stream</h2>
                        <div id="log-box" class="flex-1 bg-gray-900 p-4 rounded-lg overflow-y-auto font-mono text-xs text-green-400 space-y-1 select-text">
                            Waiting for configuration execution parameters...
                        </div>
                    </div>

                    <div class="bg-gray-800/40 p-5 rounded-xl border border-gray-800 flex flex-col h-[520px]">
                        <div class="flex justify-between items-center mb-3">
                            <h2 id="preview-title" class="text-sm font-semibold text-gray-300 uppercase tracking-wider">Report Content Preview</h2>
                            <div class="flex items-center gap-3">
                                <button id="cite-toggle-btn" onclick="openCitePanel()" disabled class="text-xs text-indigo-400 hover:underline disabled:text-gray-600 disabled:no-underline disabled:cursor-not-allowed">
                                    Cite
                                </button>
                                <button id="chat-toggle-btn" onclick="toggleChatPanel()" disabled class="text-xs text-indigo-400 hover:underline disabled:text-gray-600 disabled:no-underline disabled:cursor-not-allowed">
                                    Chat about this paper
                                </button>
                            </div>
                        </div>
                        <div id="preview-box" class="flex-1 bg-gray-900 p-4 rounded-lg overflow-y-auto font-mono text-xs text-gray-300 whitespace-pre-wrap select-text">
Select a summary sheet row from the Output Folder index to preview the extracted intelligence matrices.
                        </div>
                    </div>

                </div>
            </main>
        </div>

        <div id="chat-backdrop" onclick="closeChatPanel()" class="hidden fixed inset-0 bg-black/50 z-40 transition-opacity"></div>

        <div id="chat-panel" class="fixed top-0 right-0 h-full w-full max-w-md bg-gray-900 border-l border-gray-800 shadow-2xl z-50 flex flex-col p-5 transform translate-x-full transition-transform duration-300">
            <div class="flex justify-between items-center mb-3">
                <h2 class="text-sm font-semibold text-gray-300 uppercase tracking-wider">Chat: <span id="chat-panel-filename" class="text-indigo-400"></span></h2>
                <div class="flex items-center gap-3">
                    <button onclick="resetChatSession()" class="text-xs text-rose-400 hover:underline">Reset conversation</button>
                    <button onclick="closeChatPanel()" class="text-gray-400 hover:text-white text-lg leading-none">&times;</button>
                </div>
            </div>
            <div id="chat-log" class="flex-1 bg-gray-900 p-4 rounded-lg overflow-y-auto text-xs space-y-3 select-text">
            </div>
            <div class="flex gap-2 mt-3">
                <input id="chat-input" type="text" placeholder="Ask a question about this paper..." onkeydown="if(event.key==='Enter') sendChatMessage();" class="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-indigo-500" />
                <button id="chat-send-btn" onclick="sendChatMessage()" class="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 text-white font-medium px-4 py-2 rounded-lg text-xs transition">
                    Send
                </button>
            </div>
        </div>

        <div id="search-backdrop" onclick="closeSearchPanel()" class="hidden fixed inset-0 bg-black/50 z-40 transition-opacity"></div>

        <div id="search-panel" class="fixed top-0 left-0 h-full w-full max-w-md bg-gray-900 border-r border-gray-800 shadow-2xl z-50 flex flex-col p-5 transform -translate-x-full transition-transform duration-300">
            <div class="flex justify-between items-center mb-3">
                <h2 class="text-sm font-semibold text-gray-300 uppercase tracking-wider">Search Library</h2>
                <button onclick="closeSearchPanel()" class="text-gray-400 hover:text-white text-lg leading-none">&times;</button>
            </div>
            <div class="flex gap-2 mb-3">
                <input id="search-input" type="text" placeholder="Search across every paper..." onkeydown="if(event.key==='Enter') runLibrarySearch();" class="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-indigo-500" />
                <button id="search-btn" onclick="runLibrarySearch()" class="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 text-white font-medium px-4 py-2 rounded-lg text-xs transition">
                    Search
                </button>
            </div>
            <div id="search-results" class="flex-1 bg-gray-900 p-4 rounded-lg overflow-y-auto text-xs space-y-3 select-text">
                <p class="text-gray-500">Search matches relevant excerpts across every processed paper, ranked by semantic similarity.</p>
            </div>
        </div>

        <div id="compare-backdrop" onclick="closeComparePanel()" class="hidden fixed inset-0 bg-black/50 z-40 transition-opacity"></div>

        <div id="compare-panel" class="fixed top-0 right-0 h-full w-full max-w-md bg-gray-900 border-l border-gray-800 shadow-2xl z-50 flex flex-col p-5 transform translate-x-full transition-transform duration-300">
            <div class="flex justify-between items-center mb-3">
                <h2 class="text-sm font-semibold text-gray-300 uppercase tracking-wider">Compare Papers</h2>
                <button onclick="closeComparePanel()" class="text-gray-400 hover:text-white text-lg leading-none">&times;</button>
            </div>
            <div id="compare-results" class="flex-1 bg-gray-900 p-4 rounded-lg overflow-y-auto text-xs whitespace-pre-wrap select-text text-gray-300">
            </div>
        </div>

        <div id="cite-backdrop" onclick="closeCitePanel()" class="hidden fixed inset-0 bg-black/50 z-40 transition-opacity"></div>

        <div id="cite-panel" class="fixed top-0 right-0 h-full w-full max-w-md bg-gray-900 border-l border-gray-800 shadow-2xl z-50 flex flex-col p-5 transform translate-x-full transition-transform duration-300">
            <div class="flex justify-between items-center mb-3">
                <h2 class="text-sm font-semibold text-gray-300 uppercase tracking-wider">Cite: <span id="cite-panel-filename" class="text-indigo-400"></span></h2>
                <button onclick="closeCitePanel()" class="text-gray-400 hover:text-white text-lg leading-none">&times;</button>
            </div>
            <select id="cite-style-select" onchange="refreshCitation()" class="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-200 mb-4 focus:outline-none focus:border-indigo-500">
                <option value="apa">APA</option>
                <option value="harvard">Harvard</option>
                <option value="acl">ACL</option>
                <option value="ieee">IEEE</option>
            </select>
            <div class="mb-4">
                <div class="flex justify-between items-center mb-1">
                    <h3 class="text-xs font-semibold text-gray-400 uppercase">Formatted Citation</h3>
                    <button onclick="copyCiteText('cite-formatted')" class="text-xs text-indigo-400 hover:underline">Copy</button>
                </div>
                <div id="cite-formatted" class="bg-gray-900 border border-gray-800 rounded-lg p-3 text-xs text-gray-300 whitespace-pre-wrap select-text min-h-[3rem]"></div>
            </div>
            <div class="flex-1 flex flex-col min-h-0">
                <div class="flex justify-between items-center mb-1">
                    <h3 class="text-xs font-semibold text-gray-400 uppercase">BibTeX</h3>
                    <button onclick="copyCiteText('cite-bibtex')" class="text-xs text-indigo-400 hover:underline">Copy</button>
                </div>
                <pre id="cite-bibtex" class="flex-1 bg-gray-900 border border-gray-800 rounded-lg p-3 text-xs text-gray-300 whitespace-pre-wrap overflow-y-auto select-text"></pre>
            </div>
            <p class="text-xs text-gray-500 mt-3">BibTeX also imports directly into Zotero.</p>
        </div>

        <script>
            let dynamicActiveFile = null;
            let compareSelection = new Set();
            let metadataCache = {};

            function escapeHtml(str) {
                const div = document.createElement('div');
                div.textContent = str;
                return div.innerHTML;
            }

            async function loadMetadata() {
                try {
                    const res = await fetch('/api/metadata');
                    metadataCache = await res.json();
                } catch (err) {
                    // non-fatal - falls back to raw filenames
                }
            }

            async function uploadFile() {
                const fileSelector = document.getElementById('file-uploader');
                const feedback = document.getElementById('upload-feedback');
                if (fileSelector.files.length === 0) return;

                const targetData = new FormData();
                targetData.append('file', fileSelector.files[0]);

                feedback.innerText = "Uploading data file chunk...";
                try {
                    const res = await fetch('/api/upload', { method: 'POST', body: targetData });
                    if(res.ok) {
                        feedback.className = "text-xs text-emerald-400 mt-2 font-medium";
                        feedback.innerText = "Staged: " + fileSelector.files[0].name;
                        refreshStatus();
                    } else {
                        feedback.className = "text-xs text-rose-400 mt-2 font-medium";
                        feedback.innerText = "Staging upload failure encountered.";
                    }
                } catch (err) {
                    feedback.innerText = "Network pipeline connectivity exception.";
                }
            }

            async function uploadFromUrl() {
                const urlInput = document.getElementById('url-input');
                const url = urlInput.value.trim();
                if (!url) return;

                const feedback = document.getElementById('upload-feedback');
                const urlBtn = document.getElementById('url-upload-btn');
                feedback.className = "text-xs text-indigo-400 mt-2 font-medium";
                feedback.innerText = "Fetching URL...";
                urlInput.disabled = true;
                urlBtn.disabled = true;

                try {
                    const res = await fetch('/api/upload-url', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ url })
                    });
                    const data = await res.json();
                    if (res.ok) {
                        feedback.className = "text-xs text-emerald-400 mt-2 font-medium";
                        feedback.innerText = "Staged: " + data.filename;
                        urlInput.value = '';
                        refreshStatus();
                    } else {
                        feedback.className = "text-xs text-rose-400 mt-2 font-medium";
                        feedback.innerText = "Error: " + data.detail;
                    }
                } catch (err) {
                    feedback.className = "text-xs text-rose-400 mt-2 font-medium";
                    feedback.innerText = "Network error fetching URL.";
                } finally {
                    urlInput.disabled = false;
                    urlBtn.disabled = false;
                }
            }

            async function refreshStatus() {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                const badge = document.getElementById('status-badge');
                badge.innerText = data.status;
                if(data.status === 'Processing') {
                    badge.className = "px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 animate-pulse";
                    document.getElementById('run-btn').disabled = true;
                } else {
                    badge.className = "px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400";
                    document.getElementById('run-btn').disabled = false;
                }

                if (data.logs.length > 0) {
                    document.getElementById('log-box').innerHTML = data.logs.map(log => `<div>&gt; ${log}</div>`).join('');
                }
            }

            async function runPipeline() {
                document.getElementById('upload-feedback').innerText = "";
                await fetch('/api/run', { method: 'POST' });
                refreshStatus();
            }

            function toggleCompareSelection(baseName, checked) {
                if (checked) {
                    compareSelection.add(baseName);
                } else {
                    compareSelection.delete(baseName);
                }
                updateCompareButton();
            }

            function updateCompareButton() {
                const btn = document.getElementById('compare-btn');
                const count = compareSelection.size;
                btn.innerText = `Compare Selected (${count})`;
                btn.classList.toggle('hidden', count === 0);
                btn.disabled = count < 2;
            }

            async function refreshResultsIndex() {
                await loadMetadata();
                const res = await fetch('/api/results');
                const data = await res.json();
                const container = document.getElementById('results-list');

                if(data.files.length === 0) {
                    container.innerHTML = "No summary data instances recorded.";
                    updateCompareButton();
                    return;
                }

                container.innerHTML = data.files.map(file => {
                    const baseName = file.replace(/-summary\\.md$/, '');
                    const activeStyle = (dynamicActiveFile === file) ? 'border-indigo-500 bg-indigo-600/10 text-white' : 'border-gray-700 bg-gray-900 text-gray-300 hover:border-gray-600';
                    const meta = metadataCache[baseName];
                    const label = escapeHtml((meta && meta.title) ? meta.title : file);
                    const checked = compareSelection.has(baseName) ? 'checked' : '';
                    return `<div class="flex items-center gap-2 p-2 border rounded-lg transition text-xs ${activeStyle}">
                        <input type="checkbox" onclick="event.stopPropagation(); toggleCompareSelection('${baseName}', this.checked)" ${checked} class="shrink-0 accent-indigo-500" />
                        <div onclick="previewReport('${file}')" class="flex-1 min-w-0 cursor-pointer truncate">📄 ${label}</div>
                    </div>`;
                }).join('');
                updateCompareButton();
            }

            async function previewReport(filename) {
                dynamicActiveFile = filename;
                refreshResultsIndex();

                const res = await fetch(`/api/results/${filename}`);
                const data = await res.json();

                document.getElementById('preview-title').innerText = "Previewing: " + filename;
                document.getElementById('preview-box').innerText = data.content;

                const baseName = filename.replace(/-summary\\.md$/, '');

                const chatBtn = document.getElementById('chat-toggle-btn');
                chatBtn.disabled = false;
                chatBtn.dataset.sourceFilename = baseName;

                const citeBtn = document.getElementById('cite-toggle-btn');
                citeBtn.disabled = false;
                citeBtn.dataset.sourceFilename = baseName;

                closeChatPanel();
                closeCitePanel();
            }

            function chatBubble(role, content) {
                const isUser = role === 'user';
                const align = isUser ? 'items-end' : 'items-start';
                const bubbleStyle = isUser ? 'bg-indigo-600 text-white' : 'bg-gray-800 text-gray-200';
                const outer = document.createElement('div');
                outer.className = `flex flex-col ${align}`;
                const bubble = document.createElement('div');
                bubble.className = `max-w-[85%] px-3 py-2 rounded-lg whitespace-pre-wrap ${bubbleStyle}`;
                bubble.textContent = content;
                outer.appendChild(bubble);
                return outer;
            }

            function isChatPanelOpen() {
                return !document.getElementById('chat-panel').classList.contains('translate-x-full');
            }

            function closeChatPanel() {
                document.getElementById('chat-panel').classList.add('translate-x-full');
                document.getElementById('chat-backdrop').classList.add('hidden');
            }

            async function openChatPanel() {
                const chatBtn = document.getElementById('chat-toggle-btn');
                const sourceFilename = chatBtn.dataset.sourceFilename;
                if (!sourceFilename) return;

                closeComparePanel();
                closeCitePanel();
                document.getElementById('chat-panel').classList.remove('translate-x-full');
                document.getElementById('chat-backdrop').classList.remove('hidden');

                document.getElementById('chat-panel-filename').innerText = sourceFilename;
                const chatLog = document.getElementById('chat-log');
                chatLog.innerHTML = '';

                const res = await fetch(`/api/chat/${encodeURIComponent(sourceFilename)}/history`);
                const data = await res.json();
                data.messages.forEach(m => chatLog.appendChild(chatBubble(m.role, m.content)));
                chatLog.scrollTop = chatLog.scrollHeight;
            }

            function toggleChatPanel() {
                if (isChatPanelOpen()) {
                    closeChatPanel();
                } else {
                    openChatPanel();
                }
            }

            async function sendChatMessage() {
                const chatBtn = document.getElementById('chat-toggle-btn');
                const sourceFilename = chatBtn.dataset.sourceFilename;
                if (!sourceFilename) return;

                const input = document.getElementById('chat-input');
                const message = input.value.trim();
                if (!message) return;

                const chatLog = document.getElementById('chat-log');
                const sendBtn = document.getElementById('chat-send-btn');

                chatLog.appendChild(chatBubble('user', message));
                chatLog.scrollTop = chatLog.scrollHeight;
                input.value = '';
                input.disabled = true;
                sendBtn.disabled = true;

                try {
                    const res = await fetch(`/api/chat/${encodeURIComponent(sourceFilename)}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message })
                    });
                    if (res.ok) {
                        const data = await res.json();
                        chatLog.appendChild(chatBubble('assistant', data.content));
                    } else {
                        const err = await res.json();
                        chatLog.appendChild(chatBubble('assistant', 'Error: ' + err.detail));
                    }
                } catch (err) {
                    chatLog.appendChild(chatBubble('assistant', 'Network error contacting chat endpoint.'));
                } finally {
                    input.disabled = false;
                    sendBtn.disabled = false;
                    chatLog.scrollTop = chatLog.scrollHeight;
                    input.focus();
                }
            }

            async function resetChatSession() {
                const chatBtn = document.getElementById('chat-toggle-btn');
                const sourceFilename = chatBtn.dataset.sourceFilename;
                if (!sourceFilename) return;

                await fetch(`/api/chat/${encodeURIComponent(sourceFilename)}`, { method: 'DELETE' });
                document.getElementById('chat-log').innerHTML = '';
            }

            function isSearchPanelOpen() {
                return !document.getElementById('search-panel').classList.contains('-translate-x-full');
            }

            function closeSearchPanel() {
                document.getElementById('search-panel').classList.add('-translate-x-full');
                document.getElementById('search-backdrop').classList.add('hidden');
            }

            function openSearchPanel() {
                document.getElementById('search-panel').classList.remove('-translate-x-full');
                document.getElementById('search-backdrop').classList.remove('hidden');
                document.getElementById('search-input').focus();
            }

            function toggleSearchPanel() {
                if (isSearchPanelOpen()) {
                    closeSearchPanel();
                } else {
                    openSearchPanel();
                }
            }

            async function runLibrarySearch() {
                const input = document.getElementById('search-input');
                const query = input.value.trim();
                if (!query) return;

                const resultsBox = document.getElementById('search-results');
                const searchBtn = document.getElementById('search-btn');
                resultsBox.innerHTML = '<p class="text-gray-500">Searching across the library...</p>';
                input.disabled = true;
                searchBtn.disabled = true;

                try {
                    const res = await fetch(`/api/library-search?q=${encodeURIComponent(query)}`);
                    if (!res.ok) {
                        const err = await res.json();
                        resultsBox.innerHTML = `<p class="text-rose-400">Error: ${escapeHtml(err.detail)}</p>`;
                        return;
                    }
                    const data = await res.json();
                    if (data.results.length === 0) {
                        resultsBox.innerHTML = '<p class="text-gray-500">No matches found.</p>';
                        return;
                    }
                    resultsBox.innerHTML = data.results.map(r => {
                        const meta = metadataCache[r.source];
                        const title = escapeHtml((meta && meta.title) ? meta.title : r.source);
                        const snippet = r.excerpt.length > 400 ? r.excerpt.slice(0, 400) + '…' : r.excerpt;
                        return `<div class="p-3 bg-gray-800/60 border border-gray-700 rounded-lg">
                            <div class="flex justify-between items-center gap-2 mb-1">
                                <span class="text-indigo-400 font-semibold truncate">${title}</span>
                                <span class="text-gray-500 shrink-0">${r.score}</span>
                            </div>
                            <p class="text-gray-400 whitespace-pre-wrap">${escapeHtml(snippet)}</p>
                        </div>`;
                    }).join('');
                } catch (err) {
                    resultsBox.innerHTML = '<p class="text-rose-400">Network error contacting search endpoint.</p>';
                } finally {
                    input.disabled = false;
                    searchBtn.disabled = false;
                }
            }

            function closeComparePanel() {
                document.getElementById('compare-panel').classList.add('translate-x-full');
                document.getElementById('compare-backdrop').classList.add('hidden');
            }

            async function openComparePanel() {
                if (compareSelection.size < 2) return;
                closeChatPanel();
                closeCitePanel();

                document.getElementById('compare-panel').classList.remove('translate-x-full');
                document.getElementById('compare-backdrop').classList.remove('hidden');

                const resultsBox = document.getElementById('compare-results');
                resultsBox.innerText = 'Comparing selected papers...';

                try {
                    const res = await fetch('/api/compare', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filenames: Array.from(compareSelection) })
                    });
                    if (res.ok) {
                        const data = await res.json();
                        resultsBox.innerText = data.comparison;
                    } else {
                        const err = await res.json();
                        resultsBox.innerText = 'Error: ' + err.detail;
                    }
                } catch (err) {
                    resultsBox.innerText = 'Network error contacting compare endpoint.';
                }
            }

            function closeCitePanel() {
                document.getElementById('cite-panel').classList.add('translate-x-full');
                document.getElementById('cite-backdrop').classList.add('hidden');
            }

            async function openCitePanel() {
                const citeBtn = document.getElementById('cite-toggle-btn');
                const sourceFilename = citeBtn.dataset.sourceFilename;
                if (!sourceFilename) return;

                closeChatPanel();
                closeComparePanel();

                document.getElementById('cite-panel').classList.remove('translate-x-full');
                document.getElementById('cite-backdrop').classList.remove('hidden');
                document.getElementById('cite-panel-filename').innerText = sourceFilename;

                await refreshCitation();
            }

            async function refreshCitation() {
                const citeBtn = document.getElementById('cite-toggle-btn');
                const sourceFilename = citeBtn.dataset.sourceFilename;
                if (!sourceFilename) return;

                const style = document.getElementById('cite-style-select').value;
                const formattedBox = document.getElementById('cite-formatted');
                const bibtexBox = document.getElementById('cite-bibtex');
                formattedBox.innerText = 'Loading...';
                bibtexBox.innerText = '';

                try {
                    const res = await fetch(`/api/citation/${encodeURIComponent(sourceFilename)}?style=${style}`);
                    if (res.ok) {
                        const data = await res.json();
                        formattedBox.innerText = data.formatted;
                        bibtexBox.innerText = data.bibtex;
                    } else {
                        const err = await res.json();
                        formattedBox.innerText = 'Error: ' + err.detail;
                    }
                } catch (err) {
                    formattedBox.innerText = 'Network error contacting citation endpoint.';
                }
            }

            async function copyCiteText(elementId) {
                const text = document.getElementById(elementId).innerText;
                try {
                    await navigator.clipboard.writeText(text);
                } catch (err) {
                    // clipboard API may be unavailable (e.g. non-HTTPS) - silently no-op
                }
            }

            setInterval(refreshStatus, 2000);
            setInterval(refreshResultsIndex, 4000);
            refreshStatus();
            refreshResultsIndex();
        </script>
    </body>
    </html>
    """