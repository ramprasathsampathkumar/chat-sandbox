# Chat Sandbox — Technical Plan

## Overview

A modular chatbot with a Gradio UI, LangChain backend, swappable models (OpenAI + local via Ollama), and a built-in Responsible AI layer. Dockerized for the app container; Ollama runs natively on the host for full GPU/Metal performance.

---

## Architecture

```
User
 │
 ▼
┌─────────────────────────────────────┐
│           Gradio UI (app.py)        │
│  - Chat interface                   │
│  - Model selector dropdown          │
│  - Guardrail toggle                 │
│  - Transparency badge per message   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│         core/chain_builder.py       │
│  LCEL: history → prompt → model     │
│        → output parser              │
└──────┬──────────────┬───────────────┘
       │              │
       ▼              ▼
┌─────────────┐  ┌──────────────────┐
│ model_      │  │ memory_manager   │
│ factory.py  │  │ .py              │
│             │  │                  │
│ OpenAI      │  │ Per-session      │
│ Ollama      │  │ BufferWindow     │
│ (via enum)  │  │ Memory (k=10)    │
└─────────────┘  └──────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│         rai/ (Responsible AI)       │
│  logger.py       → JSONL audit log  │
│  guardrails.py   → input/output     │
│  explainability  → badge builder    │
└─────────────────────────────────────┘
```

---

## Project Structure

```
chat-sandbox/
├── .env                    # API keys (gitignored)
├── .env.example            # Key template
├── .gitignore
├── requirements.txt
├── Dockerfile              # App container (Option A)
├── .dockerignore
├── PLAN.md                 # This file
│
├── app.py                  # Entry point — launches Gradio
│
├── config/
│   └── settings.py         # Model registry, defaults, guardrail config
│
├── core/
│   ├── __init__.py
│   ├── model_factory.py    # Factory → LangChain BaseChatModel
│   ├── chain_builder.py    # LCEL chain assembly
│   ├── embedding_factory.py # Embedding model switcher (Phase 3)
│   └── memory_manager.py   # Per-session chat history
│
├── rai/
│   ├── __init__.py
│   ├── logger.py           # Structured JSONL logging
│   ├── guardrails.py       # Input/output content filters
│   └── explainability.py   # Model badge builder
│
├── ui/
│   ├── __init__.py
│   └── gradio_app.py       # Gradio Blocks definition
│
├── tests/
│   ├── test_model_factory.py
│   ├── test_guardrails.py
│   └── test_chain.py
│
└── logs/                   # Auto-created at runtime
    └── .gitkeep
```

---

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| UI | gradio | 4.44.1 |
| LLM orchestration | langchain | 0.3.x |
| OpenAI provider | langchain-openai | 0.2.x |
| Ollama provider | langchain-ollama | 0.2.x |
| Embeddings (local) | sentence-transformers | 3.3.x |
| Embeddings (OpenAI) | langchain-openai | (same) |
| Vector store | chromadb | 0.5.x |
| Structured logging | structlog | 24.x |
| Content moderation | better-profanity | 0.7.x |
| Env management | python-dotenv | 1.0.x |
| Validation | pydantic-settings | 2.x |
| Tokenizer | tiktoken | 0.8.x |

---

## Model Support

### Chat Models (via `ModelProvider` enum)

| Provider | Model | Notes |
|---|---|---|
| `OPENAI_GPT4O` | gpt-4o | Requires `OPENAI_API_KEY` |
| `OPENAI_GPT35` | gpt-3.5-turbo | Cheaper, faster |
| `OLLAMA_LLAMA3` | llama3.2 | Already pulled locally |
| `OLLAMA_MISTRAL` | mistral | `ollama pull mistral` required |
| `OLLAMA_CUSTOM` | user-specified | Free-text model name in UI |

### Embedding Models (via `EmbeddingProvider` enum, Phase 3)

| Provider | Model | Notes |
|---|---|---|
| `OPENAI_SMALL` | text-embedding-3-small | ~$0.02/1M tokens |
| `LOCAL_MINILM` | all-MiniLM-L6-v2 | ~90 MB download on first use |

---

## Responsible AI Layer

### Input Guardrail (before LLM call)
- Profanity filter via `better-profanity`
- PII pattern detection (SSN, credit cards) — warns, does not block
- Prompt injection detection: flags "ignore previous instructions", "you are now", etc.
- Hard-block list for explicit jailbreak patterns (configurable in `settings.py`)

### Output Guardrail (after LLM call)
- Re-run profanity check on response
- Scan for accidental system prompt leakage
- (Phase 3) Low retrieval confidence disclaimer when RAG is active

### Audit Logging
Every turn appended to `logs/chat_logs.jsonl`:
```json
{
  "ts": "2026-04-25T12:00:00Z",
  "session_id": "uuid4",
  "model_provider": "OPENAI_GPT4O",
  "model_name": "gpt-4o",
  "turn": 3,
  "input": "...",
  "output": "...",
  "input_tokens": 142,
  "output_tokens": 87,
  "latency_ms": 1243,
  "guardrail_input_flagged": false,
  "guardrail_output_flagged": false
}
```

### Transparency Badge
Appended to every assistant message:
```
[Model: gpt-4o | Provider: OpenAI | Latency: 1.2s | Tokens: 142 in / 87 out | Guardrails: ON]
```

### System Prompt (always injected, cannot be user-overridden)
```
You are a helpful assistant running as {model_name} via {provider}.
Be transparent when uncertain. Do not fabricate citations or facts.
```

---

## Docker Setup (Option A)

Ollama runs natively on the Mac host for full Metal GPU performance. Only the Gradio app is containerized.

```
Mac Host
├── Ollama (native)  ←──── Docker container talks to host.docker.internal:11434
└── Docker
    └── app container (Gradio + LangChain)
```

### Dockerfile
- Base: `python:3.12-slim` (avoids Python 3.14 wheel issues)
- Copies app code, installs requirements
- Exposes port 7860 (Gradio default)
- Env vars injected at `docker run` time via `--env-file .env`

### Run command
```bash
docker build -t chat-sandbox .
docker run --env-file .env -p 7860:7860 chat-sandbox
```

---

## Build Phases

### Phase 1 — MVP ✅ (current)
- [x] Project scaffold
- [x] `config/settings.py`
- [x] `core/model_factory.py` — OpenAI + Ollama
- [x] `core/memory_manager.py`
- [x] `core/chain_builder.py` — LCEL chain
- [x] `ui/gradio_app.py` — chat UI with model switcher
- [x] `app.py` entry point
- [x] `Dockerfile` + `.dockerignore`
- [ ] Smoke test: chat with OpenAI and Ollama, switch models mid-session

### Phase 2 — Responsible AI Layer
- [ ] `rai/logger.py`
- [ ] `rai/guardrails.py`
- [ ] `rai/explainability.py`
- [ ] Wire into Gradio respond function
- [ ] System prompt transparency text

### Phase 3 — Embeddings / RAG
- [ ] `core/embedding_factory.py`
- [ ] ChromaDB integration
- [ ] Document upload tab in Gradio
- [ ] `ConversationalRetrievalChain`
- [ ] Retrieval confidence in badge

### Phase 4 — Polish
- [ ] Custom Ollama model free-text input
- [ ] Session export (download chat as JSON)
- [ ] `ConversationSummaryBufferMemory` option
- [ ] Test suite

---

## Known Gotchas

| Issue | Mitigation |
|---|---|
| Python 3.14 wheel gaps | Dockerfile uses `python:3.12-slim`; local dev venv also targets 3.12 |
| Ollama not running | Startup health check pings `localhost:11434`; clear error with fix instructions |
| Embedding model mismatch | ChromaDB collections store embedding model metadata; factory blocks cross-model queries |
| System prompt injection | Trusted template strings only; user input never f-stringed into system prompt |
| Gradio history vs LangChain memory drift | LangChain memory is source of truth; Gradio history is display-only |
