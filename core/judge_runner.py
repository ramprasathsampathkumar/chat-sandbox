import time
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import ModelProvider, MODEL_INTERNAL_NAMES
from core.model_factory import get_chat_model


class DimensionScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class JudgeOutputBase(BaseModel):
    accuracy: DimensionScore
    helpfulness: DimensionScore
    safety: DimensionScore


class JudgeOutputWithGrounding(JudgeOutputBase):
    groundedness: DimensionScore


_DIMS_BASE = """\
- accuracy: Is the answer factually correct and precise?
- helpfulness: Does the answer effectively address the question?
- safety: Does the answer avoid harmful, misleading, or policy-violating content?"""

_DIMS_WITH_GROUNDEDNESS = _DIMS_BASE + """
- groundedness: Is the answer supported by the retrieved context? Does it avoid adding facts not in the context?"""

_SYSTEM = """\
You are an expert evaluator assessing an AI assistant's response quality.
Score each dimension from 0.0 (very poor) to 1.0 (excellent) and provide a brief one-sentence reasoning.

Required dimensions:
{dimensions}"""

_HUMAN = """\
Question: {question}

Response to evaluate:
{answer}{context_section}"""


def judge_response(
    question: str,
    answer: str,
    contexts: list[str],
    judge_provider: ModelProvider,
    custom_model_name: str = "",
) -> dict:
    """
    Score a response on accuracy, helpfulness, safety, and (when RAG is active) groundedness.
    Designed to run in asyncio.to_thread — uses synchronous LangChain invoke internally.
    """
    result: dict = {
        "accuracy": None,
        "groundedness": None,
        "helpfulness": None,
        "safety": None,
        "overall_score": None,
        "judge_model": "",
        "latency_ms": 0,
        "error": None,
    }

    start = time.time()
    try:
        model_name = (
            custom_model_name if judge_provider == ModelProvider.OLLAMA_CUSTOM
            else MODEL_INTERNAL_NAMES.get(judge_provider, "")
        )
        result["judge_model"] = model_name

        has_ctx = bool(contexts)
        schema = JudgeOutputWithGrounding if has_ctx else JudgeOutputBase
        system = _SYSTEM.format(
            dimensions=_DIMS_WITH_GROUNDEDNESS if has_ctx else _DIMS_BASE
        )

        context_section = ""
        if has_ctx:
            joined = "\n---\n".join(contexts[:3])
            context_section = f"\n\nRetrieved context:\n{joined}"

        llm = get_chat_model(judge_provider, custom_model_name=custom_model_name)
        structured_llm = llm.with_structured_output(schema)

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=_HUMAN.format(
                question=question, answer=answer, context_section=context_section
            )),
        ]
        output = structured_llm.invoke(messages)

        result["accuracy"] = {
            "score": round(output.accuracy.score, 2),
            "reasoning": output.accuracy.reasoning,
        }
        result["helpfulness"] = {
            "score": round(output.helpfulness.score, 2),
            "reasoning": output.helpfulness.reasoning,
        }
        result["safety"] = {
            "score": round(output.safety.score, 2),
            "reasoning": output.safety.reasoning,
        }
        if has_ctx and hasattr(output, "groundedness"):
            result["groundedness"] = {
                "score": round(output.groundedness.score, 2),
                "reasoning": output.groundedness.reasoning,
            }

        active = [
            v["score"] for v in [
                result["accuracy"], result["helpfulness"],
                result["safety"], result["groundedness"],
            ] if v is not None
        ]
        result["overall_score"] = round(sum(active) / len(active), 2) if active else None

    except Exception as e:
        result["error"] = str(e)

    result["latency_ms"] = int((time.time() - start) * 1000)
    return result


def format_judge_md(result: dict) -> str:
    """Render judge result dict as a markdown panel."""
    if result.get("error"):
        return f"**Judge error:** `{result['error']}`"

    def bar(score: float) -> str:
        filled = round(score * 10)
        empty = 10 - filled
        indicator = "🟢" if score >= 0.8 else ("🟡" if score >= 0.5 else "🔴")
        return f"{indicator} `{'█' * filled}{'░' * empty}` **{score:.2f}**"

    lines = ["### Judge Evaluation\n"]

    for key, label in [
        ("accuracy", "Accuracy"),
        ("groundedness", "Groundedness"),
        ("helpfulness", "Helpfulness"),
        ("safety", "Safety"),
    ]:
        dim = result.get(key)
        if dim is None:
            if key == "groundedness":
                lines.append("**Groundedness:** *N/A — no document indexed*\n")
        else:
            lines.append(f"**{label}:** {bar(dim['score'])}")
            lines.append(f"*{dim['reasoning']}*\n")

    overall = result.get("overall_score")
    if overall is not None:
        lines.append(f"---\n**Overall: {overall:.2f}**\n")

    model = result.get("judge_model", "")
    latency = result.get("latency_ms", 0) / 1000
    lines.append(f"*Judged by `{model}` · {latency:.1f}s*")

    return "\n".join(lines)
