import os
import shutil
import gc
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse
import pypdf
from smolagents import CodeAgent, OpenAIModel, tool

app = FastAPI(title="Local Agent Digest Dashboard")

model_client = OpenAIModel(
    model_id="gemma4:latest", 
    api_base="http://localhost:11434/v1",
    api_key="ollama"
)

PROCESSING_STATUS = {"status": "Idle", "current_file": None, "logs": []}

# Enforce clean workspace boundaries on your D: drive
os.makedirs("./staging", exist_ok=True)
os.makedirs("./runtime_processing", exist_ok=True)
os.makedirs("./result", exist_ok=True)

# --- LOCKED-SAFE DATA RETRIEVAL TOOL ---
@tool
def read_local_article(file_name: str) -> str:
    """
    Reads the content of an isolated paper or PDF from the active execution directory.
    Args:
        file_name: The exact filename inside the active execution directory.
    """
    # FIX: Point the tool directly to the workspace where the pipeline moved the file
    active_dir = "./runtime_processing"
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
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
            return "\n".join(text_content) if text_content else "PDF appeared empty."
            
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_name}: {str(e)}"

@tool
def write_individual_digest(content: str, source_filename: str, summary_type: str) -> str:
    """
    Saves distilled technical insights to a unique file in the result directory.
    Args:
        content: The technical Markdown text or summary matrix to record.
        source_filename: The name of the original input document.
        summary_type: The high-level parsing category.
    """
    base_name = os.path.splitext(source_filename)[0]
    output_filename = f"{base_name}-summary.md"
    result_path = os.path.join("./result", output_filename)
    
    try:
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n## [{summary_type.upper()}] (Parsed from {source_filename})\n{content}\n")
        return f"Successfully generated result asset sheet: {result_path}"
    except Exception as e:
        return f"Failed to record asset output: {str(e)}"

# --- LOCK-PROOF BACKGROUND PROCESSING PIPELINE ---
def background_agent_pipeline():
    global PROCESSING_STATUS
    staging_dir = "./staging"
    run_dir = "./runtime_processing"
    
    valid_extensions = (".txt", ".md", ".pdf")
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
        
        # Instantiate a clean agent context environment
        fresh_agent = CodeAgent(
            tools=[read_local_article, write_individual_digest],
            model=model_client,
            add_base_tools=False # Protect framework from exploring parent directories
        )
        
        prompt = f"""
        You are an elite, offline research assistant.
        1. Access the isolated file '{file}' using your 'read_local_article' tool.
        2. Extrapolate dense structural summaries, architectural configurations, or core findings.
        3. Pass the markdown summary matrix directly to 'write_individual_digest'.
           - Set source_filename explicitly to '{file}'.
           - Categorize using an appropriate summary_type.
        """
        
        try:
            fresh_agent.run(prompt)
            PROCESSING_STATUS["logs"].append(f"Completed analysis matrix for {file}.")
        except Exception as e:
            PROCESSING_STATUS["logs"].append(f"Agent engine processing exception {file}: {str(e)}")
        finally:
            # 3. Force Python's garbage collector to release any lingering library file locks
            del fresh_agent
            gc.collect()
            
            # 4. Evict the isolated document from runtime
            if os.path.exists(runtime_path):
                try:
                    os.remove(runtime_path)
                    PROCESSING_STATUS["logs"].append(f"Evicted active asset from runtime memory: {file}")
                except Exception as del_err:
                    PROCESSING_STATUS["logs"].append(f"Post-run disk clear warning for {file}: {str(del_err)}")
            
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
                            <input type="file" id="file-uploader" onchange="uploadFile()" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
                            <svg class="mx-auto h-12 w-12 text-gray-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
                            <p class="text-xs text-gray-400">Select or drop PDF, TXT, or MD documents here</p>
                            <p id="upload-feedback" class="text-xs text-indigo-400 mt-2 font-medium"></p>
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
                        <h2 id="preview-title" class="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">Report Content Preview</h2>
                        <div id="preview-box" class="flex-1 bg-gray-900 p-4 rounded-lg overflow-y-auto font-mono text-xs text-gray-300 whitespace-pre-wrap select-text">
Select a summary sheet row from the Output Folder index to preview the extracted intelligence matrices.
                        </div>
                    </div>
                </div>
            </main>
        </div>

        <script>
            let dynamicActiveFile = null;

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

            async function refreshResultsIndex() {
                const res = await fetch('/api/results');
                const data = await res.json();
                const container = document.getElementById('results-list');
                
                if(data.files.length === 0) {
                    container.innerHTML = "No summary data instances recorded.";
                    return;
                }

                container.innerHTML = data.files.map(file => {
                    const activeStyle = (dynamicActiveFile === file) ? 'border-indigo-500 bg-indigo-600/10 text-white' : 'border-gray-700 bg-gray-900 text-gray-300 hover:border-gray-600';
                    return `<div onclick="previewReport('${file}')" class="p-2 border rounded-lg cursor-pointer transition text-xs font-mono truncate ${activeStyle}">📄 ${file}</div>`;
                }).join('');
            }

            async function previewReport(filename) {
                dynamicActiveFile = filename;
                refreshResultsIndex();
                
                const res = await fetch(`/api/results/${filename}`);
                const data = await res.json();
                
                document.getElementById('preview-title').innerText = "Previewing: " + filename;
                document.getElementById('preview-box').innerText = data.content;
            }

            setInterval(refreshStatus, 2000);
            setInterval(refreshResultsIndex, 4000);
            refreshStatus();
            refreshResultsIndex();
        </script>
    </body>
    </html>
    """