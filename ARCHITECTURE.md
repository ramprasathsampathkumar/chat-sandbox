# Architecture — Chat Sandbox

## Why This Stack

### The constraint: Python 3.14

Python 3.14 was released before most ML ecosystem packages had pre-built wheels. Packages that compile Rust extensions via PyO3 (e.g. `tokenizers`, `tiktoken` source builds) hard-cap at Python ≤ 3.13. This ruled out:

- `sentence-transformers` — needs `tokenizers` (Rust/PyO3)
- `chromadb` — needs `onnxruntime` (no 3.14 wheel)
- `faiss-cpu` — C extension, no 3.14 wheel

### The solution: Ollama for everything local

Ollama exposes both LLM inference **and** embedding models over a local HTTP API. From Python's perspective, it's just an HTTP call — no C or Rust extensions, fully compatible with any Python version. This lets us run a completely local, no-GenAI-API stack on Python 3.14.

| Need | Chosen | Why |
|---|---|---|
| Chat LLM | `ChatOllama` (llama3.2) | Local, no API key |
| Embeddings | `OllamaEmbeddings` (nomic-embed-text) | Local HTTP call, no PyO3 deps |
| Vector store | `InMemoryVectorStore` (langchain-core) | Pure Python, zero deps, per-session |
| PDF loading | `PyPDFLoader` (pypdf) | Pure Python |
| Text splitting | `RecursiveCharacterTextSplitter` | Pure Python |

---

## RAG Pipeline

### Indexing (on PDF upload)

```mermaid
flowchart TD
    A([User uploads PDF]) --> B[PyPDFLoader\nlangchain-community]
    B --> C[List of Document objects\none per page]
    C --> D[RecursiveCharacterTextSplitter\nchunk_size=800, overlap=100]
    D --> E[Smaller overlapping chunks\nwith page metadata]
    E --> F[OllamaEmbeddings\nnomic-embed-text via localhost:11434]
    F --> G[Float vectors\n768 dimensions]
    G --> H[(InMemoryVectorStore\nper session, langchain-core)]
    H --> I([Ready — RAG mode active])

    style A fill:#d4edda,stroke:#28a745
    style I fill:#d4edda,stroke:#28a745
    style H fill:#cce5ff,stroke:#004085
    style F fill:#fff3cd,stroke:#856404
```

### Query / Retrieval (on each message)

```mermaid
flowchart TD
    A([User sends message]) --> B[OllamaEmbeddings\nembed the query]
    B --> C[InMemoryVectorStore\ncosine similarity search]
    C --> D[Top-8 chunks\nk=8 for broader coverage]
    D --> E{RAG grounding mode}
    E -->|Grounded — retrieval-only| F1[Strict prompt\nAnswer ONLY from retrieved chunks\nRefuse if not found]
    E -->|Augmented — retrieval + parametric| F2[Augmented prompt\nBlend chunks + model knowledge\nLabel each source]
    F1 --> G[Selected chat model]
    F2 --> G
    G --> H[Response + page citations]
    H --> I([Displayed in chat with RAG mode badge])

    style A fill:#d4edda,stroke:#28a745
    style I fill:#d4edda,stroke:#28a745
    style C fill:#cce5ff,stroke:#004085
    style G fill:#fff3cd,stroke:#856404
    style E fill:#f8d7da,stroke:#721c24
```

### Retrieval Tester (Document Inspector tab)

```mermaid
flowchart TD
    A([User types test question]) --> B[OllamaEmbeddings\nembed the query]
    B --> C[InMemoryVectorStore\nsimilarity_search_with_score]
    C --> D[Top-k chunks\nwith cosine similarity scores]
    D --> E([Rendered in UI\nScore · Page · Full chunk text])

    style A fill:#d4edda,stroke:#28a745
    style E fill:#d4edda,stroke:#28a745
    style C fill:#cce5ff,stroke:#004085
```

Use this to verify retrieval quality **before** committing to a full chat — if the right chunks don't surface here, the RAG response won't be accurate.

---

## Full System Overview

```mermaid
flowchart TD
    subgraph UI["Gradio UI (gradio_app.py)"]
        CHAT[Chat Tab\nmodel selector · grounding mode toggle]
        UPLOAD[Upload Document Tab\nstreaming indexing status]
        INSPECT[Document Inspector Tab\nchunk browser · charts · retrieval tester]
    end

    subgraph CORE["core/"]
        MF[model_factory.py\nget_chat_model]
        EF[embedding_factory.py\nget_embedding_model]
        CB[chain_builder.py\nbuild_chain\nbuild_rag_chain grounded/augmented]
        MM[memory_manager.py\nper-session window memory k=10]
        RM[rag_manager.py\nindex_pdf · get_retriever\nget_chunks · test_retrieval]
    end

    subgraph OLLAMA["Ollama — localhost:11434"]
        LLM[chat model\nllama3.2 · mistral · custom]
        EMB[nomic-embed-text\nembeddings]
    end

    subgraph RAI["rai/ — Responsible AI (Phase 2)"]
        LOG[logger.py]
        GRD[guardrails.py]
        EXP[explainability.py]
    end

    CHAT -->|message + grounding mode| CB
    UPLOAD -->|PDF path| RM
    INSPECT -->|test query| RM
    CB --> MF --> LLM
    RM --> EF --> EMB
    RM -->|retriever| CB
    CB --> MM
    CB --> RAI
```

---

## Chunking Strategy

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 800 tokens | Fits comfortably in llama3.2's context with 4 chunks + history |
| `chunk_overlap` | 100 tokens | Prevents cutting sentences mid-thought at boundaries |
| `separators` | `["\n\n", "\n", ". ", " "]` | Respects paragraph → sentence → word hierarchy |
| Top-k retrieval | 8 chunks | Increased from 4 — better coverage for broad queries like summarisation |

## Embedding Model

**nomic-embed-text** (274 MB) produces 768-dimension vectors. Chosen over `all-minilm` (the common sentence-transformers default) because:
- Ships with Ollama — zero additional installation
- 768 dims vs 384 dims — better recall on longer documents
- Apache 2.0 license — no usage restrictions

## Upgrade Path

When Python 3.14 wheels land for PyO3-based packages (expected ~late 2025):

```
OllamaEmbeddings  →  sentence-transformers/all-MiniLM-L6-v2  (offline, no Ollama dep)
InMemoryVectorStore  →  ChromaDB  (persistent across sessions)
```

Both are single-line swaps in `core/embedding_factory.py` and `core/rag_manager.py`.
