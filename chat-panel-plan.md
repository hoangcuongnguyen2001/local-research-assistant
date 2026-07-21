# Plan: Interactive Chat Side-Panel (Multi-Turn QA)

## Goal
Add a chat panel to the dashboard that lets the user ask follow-up questions about a
processed paper, grounded in that paper's content, using multi-turn conversation history.
This is scoped as a standalone feature — no vector store, no cross-document retrieval.
That's the natural follow-up once there's a corpus worth searching across.

## Prerequisite: stop deleting source documents

`background_agent_pipeline()` currently evicts the file from `./runtime_processing` in its
`finally` block right after the digest is written (agent.py:140-146). Once that happens,
there is nothing left to ground a chat session on except the (sometimes lossy/generic)
digest in `./result`.

**Change:** after a successful digest, move the file to a new `./library` directory instead
of deleting it, keyed by the same base filename as its digest. This is the one structural
change the rest of the plan depends on.

```
./staging            -> upload landing zone (unchanged)
./runtime_processing -> transient working copy during digest generation (unchanged)
./result             -> generated *-summary.md digests (unchanged)
./library             -> NEW: retained original documents, source for chat grounding
```

## Backend

### Session state
Single-user local app, so in-memory is fine for v1 — no need for a database:

```python
CHAT_SESSIONS: dict[str, list[dict]] = {}  # key: filename -> [{"role": "user"/"assistant", "content": str}, ...]
```

Lost on server restart — acceptable for v1, worth a one-line comment in code, not a real
limitation to solve now.

### Endpoints

- `POST /api/chat/{filename}` — body: `{"message": str}`
  1. Load source text for `filename` from `./library` (reuse the existing PDF/text-reading
     logic in `read_local_article`, factored out so both the pipeline and chat can call it).
  2. Append the user turn to `CHAT_SESSIONS[filename]`.
  3. Build the message list: a system prompt containing the paper text + the running
     conversation history + the new question.
  4. Call `model_client.generate(messages)` directly — **no `CodeAgent` needed here**. Chat
     QA doesn't need code execution or tools, just an instruction-following completion.
     (`OpenAIModel.generate(messages: list[ChatMessage|dict], stop_sequences=None, ...) -> ChatMessage`,
     confirmed via `inspect.signature` against the installed smolagents version.)
  5. Append the assistant turn to session state, return it to the client.

- `GET /api/chat/{filename}/history` — returns the stored turn list, so the panel can
  rehydrate after a page refresh.

- `DELETE /api/chat/{filename}` — clears a session (a "reset conversation" button in the UI).

### Context-stuffing caveat

For v1, the full paper text gets stuffed into the system prompt on every turn (no
chunking/retrieval). This works for typical paper lengths but:
- Watch Gemma's actual context window on the Ollama side — long papers plus growing chat
  history can exceed it silently and truncate.
- If that becomes a real problem, truncate to the first N tokens/pages before adding real
  retrieval — don't build a chunking scheme preemptively.
- This is exactly the seam where a future RAG pipeline plugs in: swap "stuff full text" for
  "retrieve top-k chunks" without changing the endpoint contract.

### Known model-quality risk

The same weak-model behavior already observed in Step 3 digests (generic/placeholder output)
will show up here too — a small local model asked to answer from a long context can drift
into vague answers. Nothing to design around yet; just don't be surprised by it, and revisit
if it makes the feature unusable.

## Frontend

Add a collapsible side panel, opened from the existing "Report Content Preview" card
(agent.py:251-256) when a result is selected — the natural trigger is "I'm looking at this
paper's digest, now let me ask about it."

- Toggle button next to `preview-title`.
- Panel: scrollable message thread (reuse the `log-box`/`preview-box` dark styling already
  in the dashboard) + a text input + send button, matching the existing vanilla-JS fetch
  pattern (`uploadFile`, `refreshStatus`, etc. — no new frontend framework needed).
- On open: `GET /api/chat/{filename}/history` to rehydrate.
- On send: `POST /api/chat/{filename}`, append both the optimistic user bubble and the
  returned assistant bubble.
- Disable input while a request is in flight (mirror the existing `run-btn` disable pattern).

## Milestones

1. Add `./library` dir + change pipeline to move (not delete) processed files there.
2. Factor `read_local_article`'s file-reading logic into a plain function usable outside the
   `@tool` decorator, so the chat endpoint can call it without going through a `CodeAgent`.
3. Add `CHAT_SESSIONS` state + the three endpoints above.
4. Wire the side panel into the dashboard HTML/JS.
5. Manual test: process a paper, open chat, ask a question answerable from the text, ask a
   follow-up that depends on the first answer (tests that history is actually being used).

## Explicitly out of scope for this pass

- Vector store / embeddings / chunked retrieval (RAG) — separate future effort once there's
  a multi-paper corpus to search across.
- Persistent (disk/DB) chat history across server restarts.
- Multi-user session isolation — this is a single-user local tool.
