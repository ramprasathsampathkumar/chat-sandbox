import asyncio
import re
import time
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

from config.settings import (
    ModelProvider,
    MODEL_DISPLAY_NAMES,
    check_ollama_health,
)
from core.model_factory import get_chat_model
from core.chain_builder import build_chain, build_rag_chain
from core.memory_manager import new_session_id, clear_memory
from core.rag_manager import index_pdf, get_retriever, get_chunks, clear_document, test_retrieval
from core.eval_runner import score_one_row, _build_evaluator
from core.judge_runner import judge_response, format_judge_md

PROVIDER_DISPLAY_TO_ENUM = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
DISPLAY_NAMES = list(MODEL_DISPLAY_NAMES.values())
DEFAULT_MODEL = MODEL_DISPLAY_NAMES[ModelProvider.OLLAMA_LLAMA3]

RAG_MODE_GROUNDED = "Grounded — retrieval-only"
RAG_MODE_AUGMENTED = "Augmented — retrieval + parametric"

_NO_RETRIEVAL_MSG = "Ask a question in the **Chat** tab to see retrieval results here."
_NO_COT_MSG = "Send a message in the **Chat** tab to see the chain of thought here."


def _provider_label(provider: ModelProvider) -> str:
    if provider in (ModelProvider.OPENAI_GPT4O, ModelProvider.OPENAI_GPT35):
        return "OpenAI"
    return "Ollama"


def _unpack_result(result) -> tuple[str, str]:
    """Extract (response_text, thinking) from a chain result.

    Handles two sources of thinking:
    - additional_kwargs['thinking']: qwen3 via Ollama with think=True
    - inline <think>…</think> tags: deepseek-r1 and similar models
    """
    if hasattr(result, "content"):
        text = result.content if isinstance(result.content, str) else str(result.content)
        think = result.additional_kwargs.get("reasoning_content", "")
    else:
        text = str(result)
        think = ""

    # Fallback: strip inline <think> tags if not already captured above
    if not think:
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        blocks = pattern.findall(text)
        text = pattern.sub("", text).strip()
        think = "\n\n".join(blocks).strip()

    return text.strip(), think


def respond(
    message: str,
    history: list,
    model_display: str,
    custom_model_name: str,
    session_id: str,
    rag_mode: str,
) -> tuple[str, list, str, dict]:
    empty_cot: dict = {}
    if not message.strip():
        return "", history, "", empty_cot

    provider = PROVIDER_DISPLAY_TO_ENUM[model_display]

    try:
        model = get_chat_model(provider, custom_model_name=custom_model_name)
    except ValueError as e:
        history.append({"role": "assistant", "content": f"Model error: {e}"})
        return "", history, "", empty_cot

    from config.settings import MODEL_INTERNAL_NAMES
    model_name = (
        custom_model_name
        if provider == ModelProvider.OLLAMA_CUSTOM
        else MODEL_INTERNAL_NAMES.get(provider, model_display)
    )
    provider_label = _provider_label(provider)

    retriever = get_retriever(session_id)
    start = time.time()

    try:
        if retriever is not None:
            grounded = (rag_mode == RAG_MODE_GROUNDED)
            chain, memory = build_rag_chain(model, model_name, provider_label,
                                            session_id, retriever, grounded=grounded)
            mode_tag = "retrieval-only" if grounded else "retrieval+parametric"
            rag_label = f" | **RAG: {mode_tag}**"
        else:
            chain, memory = build_chain(model, model_name, provider_label, session_id)
            rag_label = ""

        response = chain.invoke({"input": message})
    except Exception as e:
        response = f"Error calling model: {e}"
        rag_label = ""

    latency_ms = int((time.time() - start) * 1000)
    clean_response, think_content = _unpack_result(response)
    memory.save_context({"input": message}, {"output": clean_response})

    badge = (
        f"\n\n---\n"
        f"*Model: `{model_name}` | Provider: {provider_label} | "
        f"Latency: {latency_ms}ms{rag_label}*"
    )

    retrieved = test_retrieval(session_id, message) if retriever is not None else []
    cot_data = {
        "question": message,
        "rag_active": retriever is not None,
        "rag_mode": rag_mode,
        "retrieved": retrieved,
        "think_content": think_content,
    }

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": clean_response + badge})
    return "", history, message, cot_data


def upload_pdf(pdf_file, session_id: str):
    """Index an uploaded PDF, yielding status updates so the UI stays responsive."""
    if pdf_file is None:
        yield session_id, "No file selected.", gr.update(interactive=True)
        return

    filename = pdf_file.split("/")[-1]
    yield session_id, f"Indexing **{filename}**…", gr.update(interactive=False)

    try:
        n_chunks = index_pdf(session_id, pdf_file)
        yield session_id, f"Indexed **{filename}** — {n_chunks} chunks. RAG mode is now active.", gr.update(interactive=True)
    except Exception as e:
        yield session_id, f"Error indexing PDF: {e}", gr.update(interactive=True)


def build_inspector_data(session_id: str) -> tuple[pd.DataFrame, plt.Figure]:
    chunks = get_chunks(session_id)

    if not chunks:
        empty_df = pd.DataFrame(columns=["#", "Page", "Characters", "Preview"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax in axes:
            ax.text(0.5, 0.5, "No document loaded", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=12)
            ax.set_axis_off()
        fig.tight_layout()
        return empty_df, fig

    rows, char_counts, page_numbers = [], [], []
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 0)
        chars = len(chunk.page_content)
        preview = chunk.page_content[:120].replace("\n", " ").strip()
        if len(chunk.page_content) > 120:
            preview += "…"
        rows.append({"#": i + 1, "Page": page + 1, "Characters": chars, "Preview": preview})
        char_counts.append(chars)
        page_numbers.append(page + 1)

    df = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#f9f9f9")

    ax1.hist(char_counts, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax1.axvline(sum(char_counts) / len(char_counts), color="#DD4444",
                linestyle="--", linewidth=1.2,
                label=f"Mean: {sum(char_counts)//len(char_counts)} chars")
    ax1.set_title("Chunk Size Distribution", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Characters per chunk")
    ax1.set_ylabel("Number of chunks")
    ax1.legend(fontsize=9)
    ax1.set_facecolor("#ffffff")
    ax1.spines[["top", "right"]].set_visible(False)

    page_counts = Counter(page_numbers)
    pages_sorted = sorted(page_counts.keys())
    counts_sorted = [page_counts[p] for p in pages_sorted]
    bar_colors = ["#4C72B0" if c < max(counts_sorted) else "#DD4444" for c in counts_sorted]
    ax2.bar(pages_sorted, counts_sorted, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_title("Chunks per Page", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Page number")
    ax2.set_ylabel("Number of chunks")
    ax2.set_facecolor("#ffffff")
    ax2.spines[["top", "right"]].set_visible(False)

    peak_page = max(page_counts, key=page_counts.get)
    ax2.annotate(
        f"peak p.{peak_page}",
        xy=(peak_page, page_counts[peak_page]),
        xytext=(peak_page, page_counts[peak_page] + 0.3),
        ha="center", fontsize=8, color="#DD4444",
    )

    fig.tight_layout(pad=2.0)
    return df, fig


def render_retrieval_inspector(question: str, session_id: str) -> str:
    if not question:
        return _NO_RETRIEVAL_MSG
    results = test_retrieval(session_id, question)
    if not results:
        return "No document indexed — upload a PDF in the **Upload Document** tab first."

    lines = [f"### Retrieval results for: *{question}*\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"---\n"
            f"**#{i} &nbsp;|&nbsp; Page {r['page']} &nbsp;|&nbsp; Score: `{r['score']}`**\n\n"
            f"{r['text']}\n"
        )
    return "\n".join(lines)


def render_cot(cot_data: dict) -> str:
    if not cot_data:
        return _NO_COT_MSG

    question = cot_data.get("question", "")
    think = cot_data.get("think_content", "")

    if think:
        return f"## Reasoning for: *{question}*\n\n```\n{think}\n```"

    return (
        f"## Reasoning for: *{question}*\n\n"
        "*No thinking blocks detected. "
        "Select **Qwen3 8b (Ollama) 🧠** or `deepseek-r1` from the model dropdown "
        "to see chain of thought.*"
    )


_JUDGE_LOADING_MD = """
<div style="padding:4px 0">
<p style="color:#6b7280;font-style:italic;margin:0 0 8px 0">⏳ Evaluating response…</p>
<div style="background:#e5e7eb;border-radius:9999px;height:6px;overflow:hidden">
  <div style="background:linear-gradient(90deg,#10b981,#059669);height:100%;width:40%;
              border-radius:9999px;animation:judge-slide 1.2s ease-in-out infinite alternate">
  </div>
</div>
</div>
<style>@keyframes judge-slide{from{margin-left:0}to{margin-left:60%}}</style>
"""

_JUDGE_PLACEHOLDER = "Send a message to see judge results."


async def run_judge_ui(
    cot_data: dict,
    chatbot: list,
    judge_model_display: str,
    judge_custom_model: str,
):
    """Evaluate the latest response via LLM-as-Judge — runs independently of the chat chain."""
    if not cot_data:
        yield _JUDGE_PLACEHOLDER
        return

    question = cot_data.get("question", "")
    retrieved = cot_data.get("retrieved", [])
    contexts = [r["text"] for r in retrieved]

    answer = ""
    for msg in reversed(chatbot):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text") or c.get("value") or str(c) if isinstance(c, dict) else str(c)
                    for c in content
                )
            answer = content.split("\n\n---\n")[0].strip()
            break

    if not question or not answer:
        yield _JUDGE_PLACEHOLDER
        return

    yield _JUDGE_LOADING_MD

    judge_provider = PROVIDER_DISPLAY_TO_ENUM[judge_model_display]
    result = await asyncio.to_thread(
        judge_response, question, answer, contexts, judge_provider, judge_custom_model
    )
    yield format_judge_md(result)


def _parse_eval_csv(filepath: str) -> list[dict]:
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "question" not in df.columns:
        raise ValueError(f"CSV must have a 'question' column. Found: {list(df.columns)}")
    rows = []
    for _, row in df.iterrows():
        q = str(row["question"]).strip()
        if not q or q.lower() == "nan":
            continue
        gt_raw = str(row.get("ground_truth", "")).strip()
        gt = gt_raw if gt_raw and gt_raw.lower() != "nan" else None
        rows.append({"question": q, "ground_truth": gt})
    return rows


def _get_answer_and_contexts(
    question: str,
    doc_session_id: str,
    provider: ModelProvider,
    custom_model_name: str,
    rag_mode: str,
) -> tuple[str, list[str]]:
    """Run the chain for one question; uses a fresh memory session to avoid polluting chat."""
    from config.settings import MODEL_INTERNAL_NAMES
    from core.memory_manager import new_session_id as _new_sid

    model = get_chat_model(provider, custom_model_name=custom_model_name)
    model_name = (
        custom_model_name if provider == ModelProvider.OLLAMA_CUSTOM
        else MODEL_INTERNAL_NAMES.get(provider, "")
    )
    provider_label = _provider_label(provider)
    eval_sid = _new_sid()  # isolated memory — doesn't touch chat history
    retriever = get_retriever(doc_session_id)

    if retriever is not None:
        grounded = rag_mode == RAG_MODE_GROUNDED
        chain, _ = build_rag_chain(model, model_name, provider_label,
                                   eval_sid, retriever, grounded=grounded)
    else:
        chain, _ = build_chain(model, model_name, provider_label, eval_sid)

    response = chain.invoke({"input": question})
    answer, _ = _unpack_result(response)
    contexts = [r["text"] for r in test_retrieval(doc_session_id, question)] if retriever else []
    return answer, contexts


def _results_to_df(results: list[dict]) -> pd.DataFrame:
    def _fmt(v):
        return f"{v:.2f}" if v is not None else "—"

    rows = []
    for r in results:
        q = r["question"]
        rows.append({
            "Question": q[:70] + "…" if len(q) > 70 else q,
            "Ans. Relevancy": _fmt(r.get("answer_relevancy")),
            "Faithfulness": _fmt(r.get("faithfulness")),
            "Ctx. Precision": _fmt(r.get("context_precision")),
            "Ctx. Recall": _fmt(r.get("context_recall")),
            "Ans. Correctness": _fmt(r.get("answer_correctness")),
            "Error": r.get("error") or "",
        })
    return pd.DataFrame(rows)


def _make_radar_chart(results: list[dict]) -> plt.Figure | None:
    from math import pi

    metric_defs = [
        ("answer_relevancy", "Answer\nRelevancy"),
        ("faithfulness", "Faithfulness"),
        ("context_precision", "Context\nPrecision"),
        ("context_recall", "Context\nRecall"),
        ("answer_correctness", "Answer\nCorrectness"),
    ]
    active = [
        (label, sum(vals := [r[k] for r in results if r.get(k) is not None]) / len(vals))
        for k, label in metric_defs
        if any(r.get(k) is not None for r in results)
    ]
    if len(active) < 3:
        return None

    labels = [a[0] for a in active]
    values = [a[1] for a in active]
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#f9f9f9")
    ax.plot(angles, values_plot, color="#6366f1", linewidth=2)
    ax.fill(angles, values_plot, color="#6366f1", alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="grey")
    ax.grid(color="grey", alpha=0.3)
    ax.set_title("Aggregate Eval Scores", fontsize=12, fontweight="bold", pad=15)
    for angle, val, label in zip(angles[:-1], values, labels):
        ax.annotate(
            f"{val:.2f}",
            xy=(angle, val + 0.08),
            fontsize=8, fontweight="bold", color="#4f46e5",
            ha="center", va="center",
        )
    fig.tight_layout()
    return fig


async def run_batch_eval_ui(
    csv_file,
    batch_model_display: str,
    batch_custom_model: str,
    batch_rag_mode: str,
    session_id: str,
):
    """Async generator: drives the full batch eval pipeline with incremental UI updates."""
    _EMPTY_DF = pd.DataFrame(columns=[
        "Question", "Ans. Relevancy", "Faithfulness",
        "Ctx. Precision", "Ctx. Recall", "Ans. Correctness", "Error",
    ])

    if csv_file is None:
        yield "Upload a CSV file to begin.", _EMPTY_DF, None, gr.update(interactive=True)
        return

    try:
        rows = _parse_eval_csv(csv_file)
    except Exception as e:
        yield f"CSV error: {e}", _EMPTY_DF, None, gr.update(interactive=True)
        return

    if not rows:
        yield "CSV has no valid rows.", _EMPTY_DF, None, gr.update(interactive=True)
        return

    n = len(rows)
    yield f"Building evaluator… (0 / {n} questions)", _EMPTY_DF, None, gr.update(interactive=False)

    try:
        llm, emb, eval_model = await asyncio.to_thread(_build_evaluator)
    except Exception as e:
        yield f"Evaluator error: {e}", _EMPTY_DF, None, gr.update(interactive=True)
        return

    provider = PROVIDER_DISPLAY_TO_ENUM[batch_model_display]
    results: list[dict] = []

    for i, row in enumerate(rows):
        q_short = row["question"][:60] + ("…" if len(row["question"]) > 60 else "")
        yield (
            f"**{i + 1}/{n} — Generating answer:** *{q_short}*",
            _results_to_df(results) if results else _EMPTY_DF,
            None,
            gr.update(interactive=False),
        )

        try:
            answer, contexts = await asyncio.to_thread(
                _get_answer_and_contexts,
                row["question"], session_id, provider, batch_custom_model, batch_rag_mode,
            )
        except Exception as e:
            results.append({
                "question": row["question"], "answer": "", "ground_truth": row.get("ground_truth"),
                "answer_relevancy": None, "faithfulness": None, "context_precision": None,
                "context_recall": None, "answer_correctness": None,
                "evaluator_model": eval_model, "latency_ms": 0,
                "error": f"Answer generation failed: {e}",
            })
            yield (
                f"**{i + 1}/{n} — Error:** {e}",
                _results_to_df(results),
                None,
                gr.update(interactive=False),
            )
            continue

        yield (
            f"**{i + 1}/{n} — Scoring:** *{q_short}*",
            _results_to_df(results) if results else _EMPTY_DF,
            None,
            gr.update(interactive=False),
        )

        result = await asyncio.to_thread(
            score_one_row,
            row["question"], answer, contexts, row.get("ground_truth"),
            llm, emb, eval_model,
        )
        results.append(result)
        yield (
            f"**{i + 1}/{n} done ✓**",
            _results_to_df(results),
            None,
            gr.update(interactive=False),
        )

    errors = sum(1 for r in results if r.get("error"))
    summary = f"Completed — {n} question{'s' if n != 1 else ''} evaluated"
    if errors:
        summary += f", {errors} error{'s' if errors != 1 else ''}"
    fig = _make_radar_chart(results)
    yield summary, _results_to_df(results), fig, gr.update(interactive=True)


def clear_all(session_id: str):
    clear_memory(session_id)
    clear_document(session_id)
    new_id = new_session_id()
    return [], new_id, "No document loaded.", _NO_RETRIEVAL_MSG, {}, _NO_COT_MSG, _JUDGE_PLACEHOLDER


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Chat Sandbox") as demo:
        gr.Markdown("## Chat Sandbox")
        session_id_state = gr.State(new_session_id())
        last_question_state = gr.State("")
        cot_data_state = gr.State({})

        with gr.Tabs():
            # ── Chat ──────────────────────────────────────────────────────────
            with gr.Tab("Chat"):
                with gr.Row():
                    # Left: primary chat
                    with gr.Column(scale=3):
                        model_selector = gr.Dropdown(
                            choices=DISPLAY_NAMES,
                            value=DEFAULT_MODEL,
                            label="Model",
                            interactive=True,
                        )
                        custom_model_input = gr.Textbox(
                            label="Custom Ollama model name",
                            placeholder="e.g. phi3",
                            visible=False,
                        )
                        rag_mode_radio = gr.Radio(
                            choices=[RAG_MODE_GROUNDED, RAG_MODE_AUGMENTED],
                            value=RAG_MODE_GROUNDED,
                            label="RAG grounding mode",
                        )
                        chatbot = gr.Chatbot(
                            height=440,
                            show_label=False,
                            render_markdown=True,
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Type a message and press Enter…",
                                show_label=False,
                                scale=9,
                                container=False,
                            )
                            send_btn = gr.Button("Send", scale=1, variant="primary")
                        gr.Markdown(_ollama_status_md())
                        doc_status = gr.Markdown("No document loaded.")
                        clear_btn = gr.Button("Clear chat + document", variant="secondary")

                    # Right: LLM judge panel
                    with gr.Column(scale=2):
                        gr.Markdown(
                            "### LLM Judge\n"
                            "Evaluates each response on accuracy, groundedness, helpfulness, "
                            "and safety — updates automatically after every message."
                        )
                        judge_model_selector = gr.Dropdown(
                            choices=DISPLAY_NAMES,
                            value=DEFAULT_MODEL,
                            label="Judge model",
                            interactive=True,
                        )
                        judge_custom_model = gr.Textbox(
                            label="Custom judge model name",
                            placeholder="e.g. phi3",
                            visible=False,
                        )
                        judge_md = gr.Markdown(_JUDGE_PLACEHOLDER)

            # ── Upload Document ───────────────────────────────────────────────
            with gr.Tab("Upload Document"):
                gr.Markdown(
                    "Upload a PDF to enable **RAG mode**. "
                    "The document is indexed locally using `nomic-embed-text` via Ollama — "
                    "no data leaves your machine."
                )
                pdf_input = gr.File(
                    label="PDF file",
                    file_types=[".pdf"],
                    type="filepath",
                )
                upload_btn = gr.Button("Index document", variant="primary")
                upload_status = gr.Markdown("No document loaded.")

                gr.Markdown("---\n### Document Inspector")
                gr.Markdown(
                    "Inspect chunk quality after indexing. "
                    "Auto-refreshes on upload — or click **Refresh** manually."
                )
                inspect_btn = gr.Button("Refresh inspector", variant="secondary")
                with gr.Row():
                    chunk_plot = gr.Plot(label="Chunk analytics", format="png")
                with gr.Row():
                    chunk_table = gr.Dataframe(
                        headers=["#", "Page", "Characters", "Preview"],
                        label="Chunk browser",
                        wrap=True,
                        row_count=20,
                        interactive=False,
                    )

            # ── Retrieval Inspector ───────────────────────────────────────────
            with gr.Tab("Retrieval Inspector"):
                gr.Markdown(
                    "Shows the top-k chunks the retriever surfaced for the **latest chat question** — "
                    "updates automatically after every message. "
                    "Use this to verify the right sections are being retrieved before diagnosing answer quality."
                )
                retrieval_inspector_md = gr.Markdown(_NO_RETRIEVAL_MSG)

            # ── Batch Eval ────────────────────────────────────────────────────
            with gr.Tab("Batch Eval"):
                gr.Markdown(
                    "Run a full RAGAS evaluation against the currently indexed document.\n\n"
                    "**Required CSV column:** `question`  \n"
                    "**Optional column:** `ground_truth` — unlocks Context Recall and Answer Correctness.  \n"
                    "Upload a PDF in the **Upload Document** tab first to enable RAG metrics."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        batch_csv = gr.File(
                            label="Q&A CSV",
                            file_types=[".csv"],
                            type="filepath",
                        )
                        batch_run_btn = gr.Button("Run Batch Eval", variant="primary")
                        batch_progress = gr.Markdown("Upload a CSV and click **Run Batch Eval** to start.")
                    with gr.Column(scale=1):
                        batch_model = gr.Dropdown(
                            choices=DISPLAY_NAMES,
                            value=DEFAULT_MODEL,
                            label="Answer model",
                            interactive=True,
                        )
                        batch_custom_model = gr.Textbox(
                            label="Custom Ollama model name",
                            placeholder="e.g. phi3",
                            visible=False,
                        )
                        batch_rag_mode = gr.Radio(
                            choices=[RAG_MODE_GROUNDED, RAG_MODE_AUGMENTED],
                            value=RAG_MODE_GROUNDED,
                            label="RAG grounding mode",
                        )
                batch_results_df = gr.Dataframe(
                    headers=["Question", "Ans. Relevancy", "Faithfulness",
                             "Ctx. Precision", "Ctx. Recall", "Ans. Correctness", "Error"],
                    label="Per-question scores",
                    wrap=True,
                    interactive=False,
                )
                batch_radar = gr.Plot(label="Aggregate scores", format="png")

            # ── Chain of Thought ──────────────────────────────────────────────
            with gr.Tab("Chain of Thought"):
                gr.Markdown(
                    "Shows reasoning and retrieval context for the **latest chat question** — "
                    "updates automatically after every message.\n\n"
                    "**Model reasoning** requires a thinking-capable model "
                    "(e.g. `deepseek-r1`, `qwen3` via Ollama). "
                    "**RAG pipeline context** is always shown when a document is indexed."
                )
                cot_md = gr.Markdown(_NO_COT_MSG)

        # ── Event handlers ────────────────────────────────────────────────────

        def _is_custom(model_display: str) -> gr.update:
            return gr.update(
                visible=PROVIDER_DISPLAY_TO_ENUM[model_display] == ModelProvider.OLLAMA_CUSTOM
            )

        model_selector.change(_is_custom, model_selector, custom_model_input)
        judge_model_selector.change(_is_custom, judge_model_selector, judge_custom_model)

        upload_btn.click(
            upload_pdf,
            inputs=[pdf_input, session_id_state],
            outputs=[session_id_state, upload_status, upload_btn],
        ).then(
            lambda s: s,
            inputs=[upload_status],
            outputs=[doc_status],
        )

        # auto-refresh inspector after indexing
        upload_btn.click(
            build_inspector_data,
            inputs=[session_id_state],
            outputs=[chunk_table, chunk_plot],
        )

        inspect_btn.click(
            build_inspector_data,
            inputs=[session_id_state],
            outputs=[chunk_table, chunk_plot],
        )

        clear_btn.click(
            clear_all,
            inputs=[session_id_state],
            outputs=[chatbot, session_id_state, doc_status,
                     retrieval_inspector_md, cot_data_state, cot_md, judge_md],
        )

        submit_inputs = [msg_input, chatbot, model_selector, custom_model_input,
                         session_id_state, rag_mode_radio]
        submit_outputs = [msg_input, chatbot, last_question_state, cot_data_state]

        for trigger in (msg_input.submit, send_btn.click):
            trigger(respond, submit_inputs, submit_outputs).then(
                render_retrieval_inspector,
                inputs=[last_question_state, session_id_state],
                outputs=[retrieval_inspector_md],
            ).then(
                render_cot,
                inputs=[cot_data_state],
                outputs=[cot_md],
            )

        # Judge runs independently — triggered by state change, not chained after respond.
        # This frees the send button as soon as the chat chain completes.
        cot_data_state.change(
            run_judge_ui,
            inputs=[cot_data_state, chatbot, judge_model_selector, judge_custom_model],
            outputs=[judge_md],
        )

        batch_model.change(
            lambda m: gr.update(visible=PROVIDER_DISPLAY_TO_ENUM[m] == ModelProvider.OLLAMA_CUSTOM),
            inputs=[batch_model],
            outputs=[batch_custom_model],
        )

        batch_run_btn.click(
            run_batch_eval_ui,
            inputs=[batch_csv, batch_model, batch_custom_model, batch_rag_mode, session_id_state],
            outputs=[batch_progress, batch_results_df, batch_radar, batch_run_btn],
        )

    return demo


def _ollama_status_md() -> str:
    if check_ollama_health():
        return "**Ollama:** connected"
    return "**Ollama:** not reachable — local models unavailable"
