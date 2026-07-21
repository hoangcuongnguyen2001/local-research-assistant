# Session Summary — Chat Panel, RAG, and Pipeline Reliability

Date: 2026-07-21

## 1. What we built

### Chat side panel (per `chat-panel-plan.md`)
- `./library`: pipeline now archives processed source files here instead of deleting them, so there's something to ground chat on.
- `POST/GET/DELETE /api/chat/{filename}`: multi-turn Q&A grounded in the source document.
- Persistence: chat history is written to `./chat_sessions/{filename}.json` per turn and lazily reloaded into memory, so conversations survive a server restart.
- UI: a fixed slide-in overlay panel (not the originally-planned in-grid panel, which turned out to render below the fold and was easy to miss) — toggled from "Chat about this paper" next to the report preview, with a dimmed backdrop and reset/close controls.

### Retrieval-augmented chat (RAG)
- Paper text is chunked (~200 words, 40-word overlap) and embedded with `nomic-embed-text` via Ollama's native `/api/embeddings` endpoint.
- Each paper's chunk+embedding index is cached to `./embeddings/{filename}.json`, built once per document.
- Each question is embedded and compared by cosine similarity (numpy) against the cached chunks; only the top-5 most relevant chunks are sent to the model, instead of the whole paper.

### Step 3 (digest pipeline) overhaul
- Removed the `CodeAgent`/`@tool` scaffolding entirely (dead code deleted, `smolagents` no longer imported).
- Replaced with a **map-reduce summarizer**: the paper is split into chunks, grouped into ~14K-character batches, each batch summarized independently, then all partial summaries are combined into the final structured digest in one last call.
- All generation calls (chat and digest) now go through a small `ollama_chat()` helper hitting Ollama's native `/api/chat` directly, with `num_ctx` capped at 7168 tokens — the empirically verified ceiling for keeping the model 100% GPU-resident on this machine's 6GB VRAM card.
- Pipeline progress (`Summarized batch 3/11...`) now streams into `PROCESSING_STATUS["logs"]`, which the dashboard already polls every 2s — so long runs show live progress instead of going silent.

## 2. Bugs found and fixed (roughly in the order we hit them)

| # | Bug | Root cause | Fix |
|---|---|---|---|
| 1 | Chat crashed with `Error: wrong content:...` on retried turns | `smolagents`' message-merging logic required list-style `content` blocks whenever two consecutive messages shared a role; we sent plain strings | Moot — resolved by moving off `smolagents` generation entirely (see #4) |
| 2 | "FakeAVCeleb" mis-split into "FakeA VCeleb" in extracted text, causing wrong attributions in answers | `pypdf`'s default `extract_text()` inserts spurious mid-word spaces on some PDFs (confirmed: 65 broken vs. 6 correct occurrences) | Switched to `extraction_mode="layout"` |
| 3 | Layout mode then destroyed extraction on a different (2-column ACL) paper — ~5 chars/page instead of thousands | `pypdf` layout mode handles multi-column PDFs poorly | Extract both modes per page, fall back to plain text only when layout comes back >10x shorter than plain (calibrated against real per-page ratios: normal variation is ~0.8-1.4x, catastrophic failure was ~0.001x) |
| 4 | Digest generation fabricated plausible-sounding but entirely wrong content (e.g. invented "Adaptive Gating Module") even after extraction was fixed | `smolagents`' `CodeAgent` truncates tool output to 20,000 characters before the model ever sees it (`MAX_LENGTH_TRUNCATE_CONTENT`), and the multi-step agentic tool-calling loop was fragile for a weak local model on long documents | Replaced `CodeAgent` with a direct, single-purpose completion call (chat) / map-reduce summarization (digest) — no tool-calling, no truncation pathway |
| 5 | Generation was slow (~5-10 min per long paper) | Model (8B, Q4_K_M, ~9.6GB) doesn't fit in this machine's 6GB VRAM at Ollama's default 131072-token context reservation, so it ran ~72%/28% CPU/GPU | Capped `num_ctx` at 7168 (verified via `ollama ps` to be the ceiling for 100% GPU residency); combined with map-reduce batching for the digest pipeline |

## 3. Benchmarks (real numbers, not estimates)

| Scenario | Time | Notes |
|---|---|---|
| Chat: full paper stuffed in prompt | 291.8s | Old approach, CPU-bound |
| Chat: top-5 RAG chunks | 113.5s | ~2.6x faster; one-time index build ~110s/doc, then ~2s/question retrieval overhead |
| Digest: single-shot full paper (80K chars) | 292s | CPU-bound (72%/28% CPU/GPU) |
| Digest: single-shot full paper (106K chars) | 592s | CPU-bound |
| Digest: map-reduce (80K chars, 7 batches) | 206s | 100% GPU-resident throughout |
| Single chunk summarization call, `num_ctx=4096` | 15.2s | 100% GPU |
| 8-chunk batch (3,471 tokens), `num_ctx=7168` | 18.5s | 100% GPU — the calibrated batch size used in production |
| GPU residency ceiling | ~7168 tokens | Confirmed via `ollama ps`: 100% GPU up to 7168, falls back to ~66-69%/CPU-GPU split at 8192+ |

## 4. Current state

- All three papers touched this session (`2108.05080v4` / FakeAVCeleb, `2026.acl-long.662` / FinChain) now have correct, fully-grounded digests in `./result`, regenerated through the real `/api/run` pipeline with the fixes live.
- `22701-AAAI26.LeB-PHI.pdf` was processed earlier in the session and has **not** been re-verified or regenerated with the current fixes — its digest may still reflect pre-fix behavior.
- `2502.17420v2` and `2603.23301v1` summaries predate this entire session and were never touched — unknown extraction/hallucination quality.

## 5. Explicitly out of scope / not done

- Vector store or cross-document retrieval (multi-paper search) — still future work, per the original plan.
- Chat history persistence has no invalidation: if a paper's underlying PDF changes, its `./embeddings/*.json` cache and `./chat_sessions/*.json` history won't auto-refresh.
- No multi-user session isolation (single-user local tool, as originally scoped).
- `num_ctx=7168` is a fixed global cap — a very long chat conversation (many turns) could eventually exceed it; Ollama's behavior at that point (truncation vs. error) hasn't been tested.

## 6. Suggested next steps

1. **Re-verify/regenerate the other 3 existing digests** (`22701-AAAI26.LeB-PHI`, `2502.17420v2`, `2603.23301v1`) through the fixed pipeline, since they predate every fix in this session.
2. **Extend map-reduce to use the model cascade idea** discussed earlier but not implemented: use a smaller/faster model (`gemma3:4b`, 3.3GB, fits GPU trivially) for the per-batch "map" pass, reserving `gemma4` only for the final "reduce"/combine step — likely a further meaningful speedup on top of what's already been done.
3. **Cache invalidation**: decide on a policy for `./embeddings/*.json` and `./chat_sessions/*.json` if a source PDF is ever replaced or reprocessed under the same filename.
4. **Test chat behavior at the `num_ctx=7168` ceiling** with a long-running conversation to see whether Ollama truncates gracefully or errors, and handle accordingly.
5. **Consider trimming the `smolagents` dependency** from the project entirely, since nothing in `agent.py` uses it anymore (only remains in a code comment for context).


 Why reducing context window to 7168 can make gemma4:latest fit fully into 6GB GPU? Are there theoretical basis for that, especially relating to KV cache?

Good question — let me pull the actual architecture details for this model (layer count, attention head config) so I can ground the explanation in real numbers rather than hand-waving.


Good question — I pulled the actual GGUF metadata for this model, which lets me answer with real numbers instead of hand-waving. Short version: KV cache growth is real and is why the default 131072-token context doesn't fit, but it's not the whole story for the sharp cliff between 7168 and 8192 — that part is dominated by something else. Here's the breakdown.

The general theory

Total VRAM needed ≈ model weights (fixed) + KV cache (scales with context length) + compute/activation buffers (also scales with context, less predictably) + driver overhead.

The KV cache exists because, at each generation step, attention needs every previous token's Key and Value vectors, at every layer. Unlike weights, this isn't fixed — it grows every time you add a token to the context:

KV_cache_bytes = num_ctx × Σ_layers(2 × num_kv_heads × head_dim) × bytes_per_element

The 2 is for K and V; bytes_per_element is typically 2 (fp16) for Ollama's default cache type.

What this specific model actually looks like

ollama show gemma4:latest --verbose gave me the real architecture:

- 42 transformer layers (block_count)
- 8 query heads, but only 2 KV heads (head_count_kv: 2) — this is Grouped-Query Attention (GQA): 4 query heads share each KV head, which alone cuts KV cache size 4x versus standard multi-head attention.
- Hybrid local/global attention (Gemma's signature design, confirmed by checking the actual tensor shapes per layer): most layers use a sliding window capped at 512 tokens (sliding_window: 512, key_length_swa: 256), while a 1-in-6 subset of layers (I found 7 of the 42, at layers 5, 11, 17, 23, 29, 35, 41 — confirmed by their KV weight tensors being double the size of the others) do full/global attention that actually scales with the whole context (key_length: 512).

This matters enormously: for 35 of the 42 layers, the KV cache never grows past whatever a 512-token window holds, no matter how long your context is. Only 7 layers pay the full linear cost of num_ctx.

Doing the actual math

Per token, per layer (K+V combined):
- Local/SWA layer: 2 × 2 heads × 256 dim × 2 bytes = 2,048 bytes, capped at 512 tokens → 35 layers × 2,048 × 512 ≈ 35 MB total, constant regardless of num_ctx
- Global layer: 2 × 2 heads × 512 dim × 2 bytes = 4,096 bytes/token, scales with num_ctx → 7 layers × 4,096 bytes/token

┌──────────────────┬───────────────────────┬───────────────────────┬────────────────┐
│     num_ctx      │ Global-layer KV cache │ + capped local (35MB) │ Total KV cache │
├──────────────────┼───────────────────────┼───────────────────────┼────────────────┤
│ 131072 (default) │ ~3.5 GB               │ +35MB                 │ ~3.53 GB       │
├──────────────────┼───────────────────────┼───────────────────────┼────────────────┤
│ 8192             │ ~225 MB               │ +35MB                 │ ~260 MB        │
├──────────────────┼───────────────────────┼───────────────────────┼────────────────┤
│ 7168             │ ~196 MB               │ +35MB                 │ ~231 MB        │
└──────────────────┴───────────────────────┴───────────────────────┴────────────────┘

This is genuinely revealing: at the default 131072 context, KV cache alone needs ~3.5GB — combined with the ~3.3GB of weights, that's already ~6.8GB, over the 6GB budget before compute buffers are even counted. That's the real theoretical reason the default doesn't fit, and why num_ctx=32768/16384/8192 only bought marginal GPU% improvement (their KV cache is still hundreds of MB to ~1GB — meaningful, but not the whole 6GB story).

The part pure KV-cache math doesn't explain

Going from 7168 → 8192 tokens only adds about 28MB of KV cache by this math — nowhere near enough to explain the jump we measured (3.3GB/100% GPU → 9.5GB/66%-34% CPU-GPU). That cliff has to be coming from somewhere else: llama.cpp (Ollama's inference backend) also allocates compute/activation buffers sized around num_ctx (and batch size) for things like attention score scratch space and the forward-pass compute graph — and its GPU/CPU layer-placement decision is a fairly coarse, threshold-based estimate of "does everything fit," not a byte-perfect optimizer. Once the total estimated footprint crosses available VRAM, it looks like it dumps a large block of layers back to CPU at once rather than spilling incrementally — I can't verify the exact sizing formula without digging into llama.cpp's allocator source, so I don't want to overclaim precision there, but that's the mechanism consistent with everything we measured.

Summary: GQA and Gemma's sliding-window design are the real architectural reasons this 8B model's KV cache stays manageable at all (a naive full-attention model at 131K context would need far more than 3.5GB just for KV cache) — that's the solid theoretical basis. The specific 7168/8192 cliff, though, is best explained empirically (which is exactly why we tested it directly rather than trusting a formula) rather than derived cleanly from KV-cache math alone.


❯ Okay, what needs to be done if we want the agent here to be capable to connect and work with Internet (search, reference, etc)?

Getting the agent onto the internet needs four pieces, and there's one real tension worth flagging before any of it gets built.

The pieces:
1. A search backend — you need an actual search API. Given this app's offline-first design so far, the natural options split into: SearXNG (self-hosted, no API key, aggregates other engines, stays local) vs. a hosted API (Brave Search, Tavily, SerpAPI — need a key + billing, breaks the "fully local" story this app has had).
2. Fetch + extract — given a URL from search results, fetch the page and pull out the readable content (requests + something like trafilatura), the same shape of problem as PDF text extraction, just for HTML.
3. A tool-calling loop — the model needs to decide when to search versus answer from what it already has. Good news here: ollama show gemma4:latest listed tools as a capability, meaning Ollama's native /api/chat supports structured JSON tool-calling — a more constrained format than the ```python code-block parsing that CodeAgent used and that we just ripped out for being fragile. I'd build search as an Ollama-native tool call, not resurrect smolagents.
4. Prompt-injection handling — fetched web content is untrusted. It needs to be treated as data to summarize, never as instructions, the same principle I apply to tool results myself. Worth designing in from the start (clearly delimited in the prompt, like the paper excerpts already are) rather than bolted on later.

The tension: search inherently needs a decide-then-act loop, which is exactly the pattern (agentic tool-calling) whose unreliability caused most of this session's rmat is untested by us and may hold upbetter than CodeAgent's code-parsing did, but that's a real open question, not a given.