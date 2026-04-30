import time

from config.settings import settings


def _build_evaluator():
    """Return (llm, embeddings, model_name) using RAGAS-native factories."""
    from ragas.llms import llm_factory
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from ragas.embeddings import LiteLLMEmbeddings

    if settings.openai_api_key:
        import openai
        async_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        llm = llm_factory("gpt-4o-mini", provider="openai", client=async_client)
        emb = RagasOpenAIEmbeddings(client=async_client, model="text-embedding-3-small")
        return llm, emb, "gpt-4o-mini"

    # Ollama fallback via LiteLLM (model prefix: "ollama/")
    llm = llm_factory(
        "ollama/llama3.2",
        provider="litellm",
        api_base=settings.ollama_base_url,
    )
    emb = LiteLLMEmbeddings(
        model="ollama/nomic-embed-text",
        api_base=settings.ollama_base_url,
    )
    return llm, emb, "llama3.2 (Ollama)"


def score_response(question: str, answer: str, contexts: list[str]) -> dict:
    """
    Run RAGAS no-reference metrics on a single response.

    Returns:
        answer_relevancy  – always computed
        faithfulness      – only when contexts are provided
        context_precision – only when contexts are provided
        evaluator_model, latency_ms, error
    """
    result: dict = {
        "answer_relevancy": None,
        "faithfulness": None,
        "context_precision": None,
        "evaluator_model": "",
        "latency_ms": 0,
        "error": None,
    }

    start = time.time()

    try:
        from ragas.metrics.collections import (
            AnswerRelevancy,
            Faithfulness,
            ContextPrecisionWithoutReference,
        )

        llm, embeddings, model_name = _build_evaluator()
        result["evaluator_model"] = model_name

        ar = AnswerRelevancy(llm=llm, embeddings=embeddings)
        result["answer_relevancy"] = round(
            float(ar.score(user_input=question, response=answer).value), 3
        )

        if contexts:
            fa = Faithfulness(llm=llm)
            result["faithfulness"] = round(
                float(fa.score(user_input=question, response=answer, retrieved_contexts=contexts).value), 3
            )

            cp = ContextPrecisionWithoutReference(llm=llm)
            result["context_precision"] = round(
                float(cp.score(user_input=question, response=answer, retrieved_contexts=contexts).value), 3
            )

    except Exception as e:
        result["error"] = str(e)

    result["latency_ms"] = int((time.time() - start) * 1000)
    return result


def score_one_row(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None,
    llm,
    embeddings,
    evaluator_model: str,
) -> dict:
    """
    Score a single Q/A row using pre-built RAGAS objects.
    Designed to run in asyncio.to_thread — score() calls asyncio.run() internally.

    Returns all five metrics where inputs permit:
      - answer_relevancy       always
      - faithfulness           requires contexts
      - context_precision      requires contexts
      - context_recall         requires contexts + ground_truth
      - answer_correctness     requires ground_truth
    """
    from ragas.metrics.collections import (
        AnswerRelevancy,
        Faithfulness,
        ContextPrecisionWithoutReference,
        ContextRecall,
        AnswerCorrectness,
    )

    result: dict = {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "answer_relevancy": None,
        "faithfulness": None,
        "context_precision": None,
        "context_recall": None,
        "answer_correctness": None,
        "evaluator_model": evaluator_model,
        "latency_ms": 0,
        "error": None,
    }

    start = time.time()
    try:
        has_ctx = bool(contexts)
        has_gt = bool(ground_truth)

        ar = AnswerRelevancy(llm=llm, embeddings=embeddings)
        result["answer_relevancy"] = round(
            float(ar.score(user_input=question, response=answer).value), 3
        )

        if has_ctx:
            fa = Faithfulness(llm=llm)
            result["faithfulness"] = round(
                float(fa.score(user_input=question, response=answer,
                               retrieved_contexts=contexts).value), 3
            )
            cp = ContextPrecisionWithoutReference(llm=llm)
            result["context_precision"] = round(
                float(cp.score(user_input=question, response=answer,
                               retrieved_contexts=contexts).value), 3
            )

        if has_gt and has_ctx:
            cr = ContextRecall(llm=llm)
            result["context_recall"] = round(
                float(cr.score(user_input=question, retrieved_contexts=contexts,
                               reference=ground_truth).value), 3
            )

        if has_gt:
            ac = AnswerCorrectness(llm=llm, embeddings=embeddings)
            result["answer_correctness"] = round(
                float(ac.score(user_input=question, response=answer,
                               reference=ground_truth).value), 3
            )

    except Exception as e:
        result["error"] = str(e)

    result["latency_ms"] = int((time.time() - start) * 1000)
    return result


def format_eval_md(scores: dict) -> str:
    """Render eval scores dict as a markdown panel."""
    if scores.get("error"):
        return f"**Eval error:** `{scores['error']}`"

    def bar(score: float | None, rag_only: bool = False) -> str:
        if score is None:
            return "*N/A — no document indexed*" if rag_only else "*N/A*"
        filled = round(score * 10)
        empty = 10 - filled
        indicator = "🟢" if score >= 0.8 else ("🟡" if score >= 0.5 else "🔴")
        return f"{indicator} `{'█' * filled}{'░' * empty}` **{score:.2f}**"

    latency_s = scores["latency_ms"] / 1000
    model = scores["evaluator_model"]

    return "\n".join([
        "### Response Eval\n",
        "| Metric | Score |",
        "|---|---|",
        f"| Answer Relevancy | {bar(scores['answer_relevancy'])} |",
        f"| Faithfulness | {bar(scores['faithfulness'], rag_only=True)} |",
        f"| Context Precision | {bar(scores['context_precision'], rag_only=True)} |",
        f"\n*Evaluated by `{model}` · {latency_s:.1f}s*",
    ])
