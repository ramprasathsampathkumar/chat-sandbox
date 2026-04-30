# Eval Experimentation Plan

## Goal

Build a hands-on lab for experimenting with the eval methods most commonly deployed in enterprise LLM systems — starting with RAG quality metrics and progressing through LLM-as-Judge, rubric scoring, pairwise comparison, and regression pipelines. Each phase produces working, observable UI so patterns can be evaluated side by side.

---

## Method Landscape

| Method | What it measures | Ground truth needed? | Enterprise use case |
|---|---|---|---|
| **RAGAS** | RAG pipeline quality — retrieval, faithfulness, relevancy | No (no-reference) | Continuous RAG monitoring |
| **LLM-as-Judge** | Response quality via rubric — accuracy, groundedness, helpfulness, safety | No | Human-in-the-loop replacement at scale |
| **G-Eval / rubric scoring** | Custom criteria with chain-of-thought scoring | No | Domain-specific policy compliance |
| **Pairwise / A/B comparison** | Which of two models/prompts produces better responses? | No | Model selection, prompt iteration |
| **Regression eval** | Did a change break quality vs. a golden dataset? | Yes | CI/CD gating before deploys |
| **Guardrails eval** | Does the response violate safety or compliance rules? | Policy definition | Regulated industry deployment |

---

## Implementation Phases

### Phase 1 — RAGAS async scoring ✅
**Deliverable:** `core/eval_runner.py`

- `score_response(question, answer, contexts)` — 3 no-reference RAGAS metrics, async
- `score_one_row(...)` — 5-metric version for batch eval (adds Context Recall + Answer Correctness when ground truth is available)
- `_build_evaluator()` — returns pre-built RAGAS LLM + embeddings objects (OpenAI or Ollama fallback)
- `format_eval_md(scores)` — renders scores as markdown with colour indicators

Tests: `tests/test_eval_runner.py` — 25 tests, covers async-client compatibility, rounding, error capture, embeddings propagation.

---

### Phase 2 — Live per-message eval panel ✅
**Deliverable:** "Enable live eval" toggle in Chat tab

- Opt-in checkbox (off by default) — adds ~5–10s latency
- Animated progress bar while RAGAS scores are computed
- Response Eval accordion auto-opens with Answer Relevancy, Faithfulness (RAG only), Context Precision (RAG only)
- Runs in `asyncio.to_thread` — chat response appears first, scores arrive independently

---

### Phase 3 — Batch eval tab ✅
**Deliverable:** Batch Eval tab with CSV upload

- Upload CSV (`question` required, `ground_truth` optional)
- Generates answers per question using the selected model + RAG settings
- Scores all 5 RAGAS metrics where inputs permit
- Incremental table updates as each question completes
- Aggregate radar chart on completion

Test CSV generated: `~/Downloads/bitcoin_eval.csv` — 15 Q&A pairs from the Bitcoin whitepaper.

---

### Phase 4 — LLM-as-Judge (next)
**Deliverable:** Judge panel in Chat tab + `core/judge_runner.py`

**Why it matters:** RAGAS metrics are statistical proxies. An LLM judge reads the actual response and returns structured qualitative feedback — closer to how a human reviewer would evaluate at enterprise scale. G-Eval, Prometheus, and OpenAI Evals all use this pattern.

#### UI change
The Chat tab becomes two columns:

```
┌──────────────────────────────┬──────────────────────────┐
│  Primary Chat                │  Judge Panel             │
│                              │                          │
│  [Model selector]            │  [Judge model selector]  │
│  [RAG mode]                  │                          │
│  ──────────────────          │  Accuracy      ████ 0.85 │
│  Chatbot window              │  Groundedness  ████ 0.90 │
│                              │  Helpfulness   ███░ 0.70 │
│  [Message input]  [Send]     │  Safety        ████ 0.95 │
│                              │                          │
│                              │  Reasoning:              │
│                              │  "The answer is correct  │
│                              │   and well-grounded …"   │
└──────────────────────────────┴──────────────────────────┘
```

#### Backend: `core/judge_runner.py`
- `judge_response(question, answer, contexts, judge_model) -> dict`
- Rubric dimensions (scored 0–1 with reasoning):
  - **Accuracy** — is the answer factually correct?
  - **Groundedness** — is the answer supported by the retrieved context? (or "N/A" if no RAG)
  - **Helpfulness** — does the answer actually address the question?
  - **Safety** — does the answer avoid harmful, misleading, or policy-violating content?
- Structured output via JSON mode or function calling — no regex parsing
- Judge model is configurable: `gpt-4o`, `gpt-4o-mini`, or local Ollama model
- Returns `{dimension: {score, reasoning}, overall_score, judge_model, latency_ms, error}`

#### Key design decisions
- Judge panel is **always visible** in the new two-column layout (not behind a toggle)
- Scores update async after each message, same as live eval
- RAGAS live eval toggle moves into the judge panel or is retired — one eval surface is enough for the chat tab
- `judge_runner.py` must use async-compatible clients (same lesson as `eval_runner.py`)

#### Tests to add
- `test_judge_runner.py` — judges `is_async`, JSON output schema, per-dimension keys, error capture
- Same pattern as `TestBuildEvaluator` — instantiate real objects with fake key, no HTTP calls

---

### Phase 5 — Custom rubrics / G-Eval style (future)
**Deliverable:** User-defined criteria in the Judge panel

- Editable rubric textarea: user defines their own criteria (e.g. "does the answer cite a specific page?", "is the tone formal?")
- Judge model applies chain-of-thought scoring to each criterion
- Saves rubric per session — can be exported for reuse

**Enterprise relevance:** Different teams need different criteria — legal, support, product, compliance. A configurable rubric is the key to making LLM-as-Judge reusable across use cases.

---

### Phase 6 — Pairwise / A/B model comparison (future)
**Deliverable:** Side-by-side model comparison tab

- Same question sent to two models simultaneously
- LLM judge declares a winner per dimension
- Aggregate win-rate table across a batch of questions
- Use case: picking between `gpt-4o` vs `llama3.2` for a specific task, or comparing prompt variants

---

### Phase 7 — Regression eval pipeline (future)
**Deliverable:** Score tracking against a golden dataset

- Store scores per run in a local SQLite or JSONL log
- Compare current scores vs. a baseline run
- Flag regressions: metric dropped by more than a threshold
- Use case: CI gating — block deploys if Answer Correctness drops below 0.75 on the golden test set

---

## Framework Comparison

| Framework | Verdict |
|---|---|
| **RAGAS** | In use — best fit for no-reference RAG metrics |
| **DeepEval** | Strong LLM-as-Judge + G-Eval support; worth importing for Phase 5 |
| **LangChain Evaluators** | Already in stack; good fallback for simple criteria scoring |
| **TruLens** | Full observability dashboard; heavier setup; worth revisiting for Phase 7 |
| **OpenAI Evals** | Hosted, API-driven; good for Phase 7 regression pipelines |

---

## Test Dataset Strategy

| Tier | What | Unlocks |
|---|---|---|
| Quick start | Question-only CSV | Answer Relevancy only |
| Standard | `question, ground_truth` CSV | All 5 RAGAS metrics |
| Adversarial | Questions designed to trip up retrieval or faithfulness | Stress-tests eval sensitivity |

Current test set: `bitcoin_eval.csv` — 15 Q&A pairs, all sections of the whitepaper covered.
