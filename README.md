# Chat Sandbox

A modular chatbot with a Gradio UI, LangChain backend, swappable models (OpenAI + local via Ollama), PDF-based RAG, and a built-in Responsible AI layer.

---

## Features

- **Swappable models** — GPT-4o, GPT-3.5-turbo, llama3.2, mistral, Qwen3 8b, or any custom Ollama model
- **Local PDF RAG** — upload a PDF, index it locally with `nomic-embed-text`, chat against it with no data leaving your machine
- **RAG grounding mode** — toggle between *Grounded (retrieval-only)* and *Augmented (retrieval + parametric)*
- **Chain of Thought** — see the model's internal reasoning for every message; full thinking blocks for Qwen3 and deepseek-r1
- **Retrieval Inspector** — auto-updated after every chat message; shows the exact chunks retrieved with similarity scores and page numbers
- **Document Inspector** — chunk browser and size distribution charts to verify indexing quality
- **Per-session memory** — 10-turn sliding window, isolated per browser session
- **Transparency badges** — every response shows model, provider, latency, and RAG mode
- **LLM-as-Judge panel** — always-visible judge panel in the chat view; scores every response on Accuracy, Groundedness, Helpfulness, and Safety (0–1 each) with one-sentence reasoning per dimension; judge model is independently configurable
- **Live RAGAS eval** — optional per-message RAGAS metrics (Answer Relevancy, Faithfulness, Context Precision) via OpenAI or Ollama; runs async so the chat response arrives first

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
ollama pull qwen3:8b          # optional — enables Chain of Thought
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

# 5. Start the app  ← always use app.py, not ui/gradio_app.py directly
python3 app.py
```

Open **http://localhost:7860** in your browser.

> **Important:** always launch via `app.py` from the project root with the venv active.
> Running `ui/gradio_app.py` directly will fail with `ModuleNotFoundError: No module named 'config'`
> because `config/` and `core/` are resolved relative to the project root.

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

1. Select a model from the dropdown. Select **Qwen3 8b (Ollama) 🧠** to see reasoning in the Chain of Thought tab.
2. Choose a **RAG grounding mode** (only applies once a document is indexed):
   - **Grounded — retrieval-only**: answers are strictly limited to the PDF chunks; refuses if the answer is not in the document
   - **Augmented — retrieval + parametric**: model may blend document content with its training knowledge, labelling each source
3. Type a message and press Enter or click **Send**

The Retrieval Inspector and Chain of Thought tabs update automatically after every message.

### Upload Document tab

1. Select a PDF file and click **Index document** — status updates in real time
2. Once indexed, RAG mode activates automatically in the Chat tab
3. The **Document Inspector** section shows chunk analytics (size distribution chart, chunks-per-page chart, full chunk browser table) — auto-refreshes after indexing, or click **Refresh inspector** manually

### Retrieval Inspector tab

Shows the top-k chunks the retriever surfaced for the **latest chat question**, updated automatically after every message. Use this to verify the right sections are being retrieved before diagnosing answer quality.

### Chain of Thought tab

Shows the model's internal reasoning for the **latest chat question**, updated automatically after every message.

- **Qwen3** (via Ollama): full thinking blocks via `reasoning=True` — the model reasons before answering
- **deepseek-r1** and similar: inline `<think>…</think>` tags are extracted automatically
- All other models: no thinking blocks; a prompt is shown to switch to a reasoning model

---

## Evaluation

The app has two independent evaluation layers that run after every chat message.

### LLM-as-Judge (always on)

A configurable judge model scores the latest response on four rubric dimensions. The judge model is selected separately from the chat model — any supported model can serve as judge.

| Dimension | What it measures | Scored when |
|---|---|---|
| **Accuracy** | Is the answer factually correct and precise? | Always |
| **Helpfulness** | Does the answer effectively address the question? | Always |
| **Safety** | Does the answer avoid harmful, misleading, or policy-violating content? | Always |
| **Groundedness** | Is the answer supported by the retrieved context? Does it avoid adding facts not in the context? | RAG mode only |

Each dimension returns a score from **0.0** (very poor) to **1.0** (excellent) plus a one-sentence reasoning. An **overall score** is the mean of all active dimensions. Results are rendered as a colour-coded bar chart (🟢 ≥ 0.8 · 🟡 ≥ 0.5 · 🔴 < 0.5).

Implementation: `core/judge_runner.py` — uses `with_structured_output` (Pydantic schema) for guaranteed JSON; no regex parsing. Runs in `asyncio.to_thread` so it never blocks the chat response.

### RAGAS per-message scoring (opt-in)

Enable the **"Enable live eval"** toggle in the Chat tab to add RAGAS scores after each message (~5–10 s added latency).

| Metric | What it measures | Requires |
|---|---|---|
| **Answer Relevancy** | Does the answer address the question? | Question + answer |
| **Faithfulness** | Does the answer stay within the retrieved context? | RAG mode |
| **Context Precision** | Are the retrieved chunks actually relevant to the question? | RAG mode |

Full 5-metric batch eval (adds Context Recall + Answer Correctness when a ground-truth column is provided) is available in the **Batch Eval** tab via CSV upload.

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
│   ├── rag_manager.py          # PDF indexing, retrieval, chunk inspection
│   └── eval_runner.py          # RAGAS per-message eval (async, no-reference metrics)
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
| LLM-as-Judge | `core/judge_runner.py` — Accuracy, Groundedness, Helpfulness, Safety |
| Eval metrics | RAGAS 0.2+ (Answer Relevancy, Faithfulness, Context Precision) |
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

Use the **Retrieval Inspector** tab after sending a question to verify chunks are being retrieved correctly.
