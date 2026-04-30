"""
Tests for core/judge_runner.py.

All tests use mocked LLM calls — no API keys or network needed.
The key seam being tested: get_chat_model → with_structured_output → invoke.
"""
import pytest
from unittest.mock import MagicMock, patch

from core.judge_runner import (
    judge_response,
    format_judge_md,
    DimensionScore,
    JudgeOutputBase,
    JudgeOutputWithGrounding,
)
from config.settings import ModelProvider


# ── helpers ───────────────────────────────────────────────────────────────────

def _dim(score=0.9, reasoning="ok"):
    return DimensionScore(score=score, reasoning=reasoning)


def _base_output(acc=0.9, hlp=0.8, saf=0.95):
    return JudgeOutputBase(
        accuracy=_dim(acc, "Accurate."),
        helpfulness=_dim(hlp, "Helpful."),
        safety=_dim(saf, "Safe."),
    )


def _grounded_output(acc=0.9, hlp=0.8, saf=0.95, gnd=0.85):
    return JudgeOutputWithGrounding(
        accuracy=_dim(acc, "Accurate."),
        helpfulness=_dim(hlp, "Helpful."),
        safety=_dim(saf, "Safe."),
        groundedness=_dim(gnd, "Grounded."),
    )


def _mock_llm(output):
    """Return a mock LLM whose with_structured_output().invoke() returns output."""
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = output
    return llm


# ── judge_response: output schema ─────────────────────────────────────────────

class TestJudgeResponseSchema:
    def test_returns_expected_keys(self):
        with patch("core.judge_runner.get_chat_model", return_value=_mock_llm(_base_output())):
            result = judge_response("Q?", "A.", [], ModelProvider.OLLAMA_LLAMA3)
        assert set(result.keys()) == {
            "accuracy", "groundedness", "helpfulness", "safety",
            "overall_score", "judge_model", "latency_ms", "error",
        }

    def test_no_contexts_groundedness_is_none(self):
        with patch("core.judge_runner.get_chat_model", return_value=_mock_llm(_base_output())):
            result = judge_response("Q?", "A.", [], ModelProvider.OLLAMA_LLAMA3)
        assert result["groundedness"] is None
        assert result["error"] is None

    def test_with_contexts_includes_groundedness(self):
        with patch("core.judge_runner.get_chat_model", return_value=_mock_llm(_grounded_output())):
            result = judge_response("Q?", "A.", ["ctx"], ModelProvider.OLLAMA_LLAMA3)
        assert result["groundedness"] is not None
        assert result["groundedness"]["score"] == 0.85

    def test_each_dimension_has_score_and_reasoning(self):
        with patch("core.judge_runner.get_chat_model", return_value=_mock_llm(_grounded_output())):
            result = judge_response("Q?", "A.", ["ctx"], ModelProvider.OLLAMA_LLAMA3)
        for key in ("accuracy", "helpfulness", "safety", "groundedness"):
            assert "score" in result[key], f"{key} missing score"
            assert "reasoning" in result[key], f"{key} missing reasoning"

    def test_scores_rounded_to_2dp(self):
        out = JudgeOutputBase(
            accuracy=_dim(0.876543),
            helpfulness=_dim(0.876543),
            safety=_dim(0.876543),
        )
        with patch("core.judge_runner.get_chat_model", return_value=_mock_llm(out)):
            result = judge_response("Q?", "A.", [], ModelProvider.OLLAMA_LLAMA3)
        assert result["accuracy"]["score"] == 0.88

    def test_overall_score_is_mean_of_active_dimensions(self):
        with patch("core.judge_runner.get_chat_model", return_value=_mock_llm(
            _base_output(acc=0.8, hlp=0.6, saf=1.0)
        )):
            result = judge_response("Q?", "A.", [], ModelProvider.OLLAMA_LLAMA3)
        expected = round((0.8 + 0.6 + 1.0) / 3, 2)
        assert result["overall_score"] == expected

    def test_overall_score_includes_groundedness_when_present(self):
        with patch("core.judge_runner.get_chat_model", return_value=_mock_llm(
            _grounded_output(acc=1.0, hlp=1.0, saf=1.0, gnd=0.0)
        )):
            result = judge_response("Q?", "A.", ["ctx"], ModelProvider.OLLAMA_LLAMA3)
        assert result["overall_score"] == round((1.0 + 1.0 + 1.0 + 0.0) / 4, 2)

    def test_judge_model_name_propagated(self):
        with patch("core.judge_runner.get_chat_model", return_value=_mock_llm(_base_output())):
            result = judge_response("Q?", "A.", [], ModelProvider.OLLAMA_LLAMA3)
        assert isinstance(result["judge_model"], str) and result["judge_model"] != ""

    def test_latency_ms_is_non_negative(self):
        with patch("core.judge_runner.get_chat_model", return_value=_mock_llm(_base_output())):
            result = judge_response("Q?", "A.", [], ModelProvider.OLLAMA_LLAMA3)
        assert result["latency_ms"] >= 0


# ── judge_response: schema selection ──────────────────────────────────────────

class TestSchemaSelection:
    """Verifies the correct Pydantic schema is passed to with_structured_output.
    Mocking with_structured_output entirely would hide this seam."""

    def _capture_schema(self, output):
        captured = {}

        def _side_effect(schema):
            captured["schema"] = schema
            m = MagicMock()
            m.invoke.return_value = output
            return m

        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = _side_effect
        return mock_llm, captured

    def test_grounded_schema_used_when_contexts_provided(self):
        mock_llm, captured = self._capture_schema(_grounded_output())
        with patch("core.judge_runner.get_chat_model", return_value=mock_llm):
            judge_response("Q?", "A.", ["ctx"], ModelProvider.OLLAMA_LLAMA3)
        assert captured["schema"] is JudgeOutputWithGrounding

    def test_base_schema_used_when_no_contexts(self):
        mock_llm, captured = self._capture_schema(_base_output())
        with patch("core.judge_runner.get_chat_model", return_value=mock_llm):
            judge_response("Q?", "A.", [], ModelProvider.OLLAMA_LLAMA3)
        assert captured["schema"] is JudgeOutputBase


# ── judge_response: error handling ────────────────────────────────────────────

class TestJudgeResponseErrors:
    def test_llm_error_captured_not_raised(self):
        bad_llm = MagicMock()
        bad_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM down")
        with patch("core.judge_runner.get_chat_model", return_value=bad_llm):
            result = judge_response("Q?", "A.", [], ModelProvider.OLLAMA_LLAMA3)
        assert result["error"] == "LLM down"
        assert result["accuracy"] is None
        assert result["overall_score"] is None

    def test_model_init_error_captured(self):
        with patch("core.judge_runner.get_chat_model", side_effect=ValueError("bad model")):
            result = judge_response("Q?", "A.", [], ModelProvider.OLLAMA_LLAMA3)
        assert "bad model" in result["error"]


# ── format_judge_md ───────────────────────────────────────────────────────────

class TestFormatJudgeMd:
    def _result(self, **overrides):
        base = {
            "accuracy": {"score": 0.9, "reasoning": "Accurate."},
            "helpfulness": {"score": 0.8, "reasoning": "Helpful."},
            "safety": {"score": 0.95, "reasoning": "Safe."},
            "groundedness": None,
            "overall_score": 0.88,
            "judge_model": "llama3.2",
            "latency_ms": 3200,
            "error": None,
        }
        return {**base, **overrides}

    def test_error_state(self):
        md = format_judge_md({
            "error": "something broke",
            "accuracy": None, "helpfulness": None, "safety": None,
            "groundedness": None, "overall_score": None,
            "judge_model": "", "latency_ms": 0,
        })
        assert "Judge error" in md
        assert "something broke" in md

    def test_scores_rendered(self):
        md = format_judge_md(self._result())
        assert "0.90" in md
        assert "0.80" in md
        assert "0.95" in md

    def test_model_and_latency_shown(self):
        md = format_judge_md(self._result())
        assert "llama3.2" in md
        assert "3.2s" in md

    def test_no_groundedness_shows_na(self):
        md = format_judge_md(self._result(groundedness=None))
        assert "N/A" in md

    def test_groundedness_shown_when_present(self):
        md = format_judge_md(self._result(
            groundedness={"score": 0.85, "reasoning": "Well grounded."}
        ))
        assert "0.85" in md
        assert "Well grounded." in md

    def test_green_amber_red_indicators(self):
        md = format_judge_md(self._result(
            accuracy={"score": 0.9, "reasoning": "r"},    # green
            helpfulness={"score": 0.6, "reasoning": "r"}, # amber
            safety={"score": 0.3, "reasoning": "r"},       # red
        ))
        assert "🟢" in md
        assert "🟡" in md
        assert "🔴" in md

    def test_overall_score_shown(self):
        md = format_judge_md(self._result(overall_score=0.88))
        assert "0.88" in md

    def test_reasoning_text_included(self):
        md = format_judge_md(self._result(
            accuracy={"score": 0.9, "reasoning": "The answer cites correct figures."}
        ))
        assert "The answer cites correct figures." in md
