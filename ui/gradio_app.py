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


def clear_all(session_id: str):
    clear_memory(session_id)
    clear_document(session_id)
    new_id = new_session_id()
    return [], new_id, "No document loaded.", _NO_RETRIEVAL_MSG, {}, _NO_COT_MSG


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
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            height=520,
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

                    with gr.Column(scale=1, min_width=220):
                        gr.Markdown("### Settings")
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
                        gr.Markdown(_ollama_status_md())
                        doc_status = gr.Markdown("No document loaded.")
                        clear_btn = gr.Button("Clear chat + document", variant="secondary")

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

        def toggle_custom(model_display: str) -> gr.update:
            return gr.update(
                visible=PROVIDER_DISPLAY_TO_ENUM[model_display] == ModelProvider.OLLAMA_CUSTOM
            )

        model_selector.change(toggle_custom, model_selector, custom_model_input)

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
                     retrieval_inspector_md, cot_data_state, cot_md],
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

    return demo


def _ollama_status_md() -> str:
    if check_ollama_health():
        return "**Ollama:** connected"
    return "**Ollama:** not reachable — local models unavailable"
