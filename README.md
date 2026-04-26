# Chat Sandbox

A modular chatbot with a Gradio UI, LangChain backend, swappable models (OpenAI + local via Ollama), PDF-based RAG, and a built-in Responsible AI layer.

---

## Features

- **Swappable models** — GPT-4o, GPT-3.5-turbo, llama3.2, mistral, or any custom Ollama model
- **Local PDF RAG** — upload a PDF, index it locally with `nomic-embed-text`, chat against it with no data leaving your machine
- **RAG grounding mode** — toggle between *Grounded (retrieval-only)* and *Augmented (retrieval + parametric)*
- **Document Inspector** — chunk browser, size distribution charts, and a retrieval tester to verify RAG quality before chatting
- **Per-session memory** — 10-turn sliding window, isolated per browser session
- **Transparency badges** — every response shows model, provider, latency, and RAG mode

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12+ | 3.14 works for all MVP features |
| [Ollama](https://ollama.com) | latest | Must be running before starting the app |
| OpenAI API key | — | Optional — only needed for GPT models |

### Pull required Ollama models

```bash
ollama pull llama3.2          # default chat model
ollama pull nomic-embed-text  # required for PDF RAG
ollama pull mistral           # optional
```

---

## Quickstart — Local Dev

```bash
# 1. Clone
git clone git@github.com:ramprasathsampathkumar/chat-sandbox.git
cd chat-sandbox

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set your values (see Configuration below)

# 5. Start the app
python3 app.py
```

Open **http://localhost:7860** in your browser.

---

## Configuration

Copy `.env.example` to `.env` and fill in:

```env
OPENAI_API_KEY=sk-...                        # Optional — leave blank to use Ollama only
OLLAMA_BASE_URL=http://localhost:11434        # Local dev
# OLLAMA_BASE_URL=http://host.docker.internal:11434  # When running inside Docker
```

---

## Quickstart — Docker

Ollama runs natively on the host for full GPU/Metal performance. Only the app is containerized.

```bash
# 1. Make sure Ollama is running on the host
ollama serve

# 2. Build the image
docker build -t chat-sandbox .

# 3. Run — uses host.docker.internal to reach Ollama
docker run --env-file .env -p 7860:7860 chat-sandbox
```

Open **http://localhost:7860**.

> `.env` must contain `OLLAMA_BASE_URL=http://host.docker.internal:11434` when running in Docker.

---

## Usage Guide

### Chat tab

1. Select a model from the dropdown
2. Choose a **RAG grounding mode** (appears once a document is indexed):
   - **Grounded — retrieval-only**: answers are strictly limited to the PDF chunks
   - **Augmented — retrieval + parametric**: model may blend document content with its training knowledge, labelling each source
3. Type a message and press Enter or click **Send**

### Upload Document tab

1. Select a PDF file
2. Click **Index document** — status updates in real time
3. Once indexed, RAG mode activates automatically in the Chat tab

### Document Inspector tab

1. Click **Load inspector** to see chunk analytics (size distribution, chunks per page)
2. Use the **Retrieval Tester** at the bottom — type a question and click **Test retrieval** to see the exact chunks the retriever would surface, with similarity scores and page numbers, before committing to a full chat

---

## Project Structure

```
chat-sandbox/
├── app.py                      # Entry point — launches Gradio
├── requirements.txt
├── Dockerfile
├── .env.example
│
├── config/
│   └── settings.py             # Model registry, enums, health checks
│
├── core/
│   ├── model_factory.py        # Returns BaseChatModel for any provider
│   ├── chain_builder.py        # LCEL chains (plain + RAG, grounded + augmented)
│   ├── memory_manager.py       # Per-session ConversationBufferWindowMemory
│   ├── embedding_factory.py    # OllamaEmbeddings factory
│   └── rag_manager.py          # PDF indexing, retrieval, chunk inspection
│
├── ui/
│   └── gradio_app.py           # Gradio Blocks — all tabs and event handlers
│
├── rai/                        # Responsible AI layer (Phase 2)
│   ├── logger.py
│   ├── guardrails.py
│   └── explainability.py
│
└── logs/                       # JSONL audit logs (gitignored)
```

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Gradio 6.x |
| LLM orchestration | LangChain 0.3.x / 1.x |
| OpenAI provider | langchain-openai |
| Ollama provider | langchain-ollama |
| Embeddings | OllamaEmbeddings (nomic-embed-text) |
| Vector store | InMemoryVectorStore (langchain-core) |
| PDF loading | PyPDFLoader (pypdf) |
| Structured logging | structlog |
| Content moderation | better-profanity |
| Config | pydantic-settings |

---

## Good PDFs for RAG Testing

Short, fact-dense documents work best:

| Document | Why it's useful |
|---|---|
| [Bitcoin Whitepaper](https://bitcoin.org/bitcoin.pdf) | 9 pages, precise technical claims, easy to verify |
| [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) | Specific numbers: BLEU scores, hyperparameters, layer counts |
| Any company privacy policy PDF | Real-world RAG use case, tests legal language retrieval |

Use the **Retrieval Tester** in the Document Inspector tab to verify chunks are being retrieved correctly before chatting.
