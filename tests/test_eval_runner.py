"""
Tests for core/eval_runner.py.

Fast tests (default): mock the RAGAS metric calls — no API keys or network needed.
Integration test (opt-in): calls real OpenAI API, skipped unless --run-integration is passed.
    pytest tests/test_eval_runner.py --run-integration
"""
import pytest
from unittest.mock import MagicMock, patch

import io
import tempfile
import os
import pandas as pd

from core.eval_runner import format_eval_md, score_response, _build_evaluator, score_one_row


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_metric_result(value: float):
    """Return a minimal MetricResult-like object."""
    r = MagicMock()
    r.value = value
    return r


def _mock_evaluator():
    """Patch _build_evaluator to return dummy objects and a known model name."""
    fake_llm = MagicMock()
    fake_emb = MagicMock()
    return patch("core.eval_runner._build_evaluator", return_value=(fake_llm, fake_emb, "mock-model"))


# ── format_eval_md ────────────────────────────────────────────────────────────

class TestFormatEvalMd:
    def test_error_state(self):
        md = format_eval_md({"error": "something broke", "answer_relevancy": None,
                              "faithfulness": None, "context_precision": None,
                              "evaluator_model": "", "latency_ms": 0})
        assert "Eval error" in md
        assert "something broke" in md

    def test_full_scores_rendered(self):
        md = format_eval_md({
            "answer_relevancy": 0.9,
            "faithfulness": 0.75,
            "context_precision": 0.4,
            "evaluator_model": "gpt-4o-mini",
            "latency_ms": 3200,
            "error": None,
        })
        assert "0.90" in md
        assert "0.75" in md
        assert "0.40" in md
        assert "gpt-4o-mini" in md
        assert "3.2s" in md

    def test_green_amber_red_indicators(self):
        md = format_eval_md({
            "answer_relevancy": 0.9,   # green
            "faithfulness": 0.6,       # amber
            "context_precision": 0.3,  # red
            "evaluator_model": "x", "latency_ms": 0, "error": None,
        })
        assert "🟢" in md
        assert "🟡" in md
        assert "🔴" in md

    def test_no_context_shows_na(self):
        md = format_eval_md({
            "answer_relevancy": 0.8,
            "faithfulness": None,
            "context_precision": None,
            "evaluator_model": "x", "latency_ms": 0, "error": None,
        })
        assert "N/A" in md


# ── score_response (mocked) ───────────────────────────────────────────────────

class TestScoreResponse:
    def test_returns_expected_keys(self):
        with _mock_evaluator():
            with patch("ragas.metrics.collections.AnswerRelevancy") as MockAR:
                MockAR.return_value.score.return_value = _make_metric_result(0.85)
                result = score_response("Q?", "A.", contexts=[])

        assert set(result.keys()) == {
            "answer_relevancy", "faithfulness", "context_precision",
            "evaluator_model", "latency_ms", "error",
        }

    def test_no_context_skips_rag_metrics(self):
        with _mock_evaluator():
            with patch("ragas.metrics.collections.AnswerRelevancy") as MockAR:
                MockAR.return_value.score.return_value = _make_metric_result(0.9)
                result = score_response("Q?", "A.", contexts=[])

        assert result["faithfulness"] is None
        assert result["context_precision"] is None
        assert result["error"] is None

    def test_with_context_scores_all_three(self):
        with _mock_evaluator():
            with patch("ragas.metrics.collections.AnswerRelevancy") as MockAR, \
                 patch("ragas.metrics.collections.Faithfulness") as MockFA, \
                 patch("ragas.metrics.collections.ContextPrecisionWithoutReference") as MockCP:
                MockAR.return_value.score.return_value = _make_metric_result(0.9)
                MockFA.return_value.score.return_value = _make_metric_result(0.8)
                MockCP.return_value.score.return_value = _make_metric_result(0.7)
                result = score_response("Q?", "A.", contexts=["chunk1", "chunk2"])

        assert result["answer_relevancy"] == 0.9
        assert result["faithfulness"] == 0.8
        assert result["context_precision"] == 0.7
        assert result["error"] is None

    def test_scores_are_rounded_to_3dp(self):
        with _mock_evaluator():
            with patch("ragas.metrics.collections.AnswerRelevancy") as MockAR:
                MockAR.return_value.score.return_value = _make_metric_result(0.856789)
                result = score_response("Q?", "A.", contexts=[])

        assert result["answer_relevancy"] == 0.857

    def test_metric_error_captured_not_raised(self):
        with _mock_evaluator():
            with patch("ragas.metrics.collections.AnswerRelevancy") as MockAR:
                MockAR.return_value.score.side_effect = RuntimeError("API timeout")
                result = score_response("Q?", "A.", contexts=[])

        assert result["error"] == "API timeout"
        assert result["answer_relevancy"] is None

    def test_evaluator_model_name_propagated(self):
        with _mock_evaluator():
            with patch("ragas.metrics.collections.AnswerRelevancy") as MockAR:
                MockAR.return_value.score.return_value = _make_metric_result(0.8)
                result = score_response("Q?", "A.", contexts=[])

        assert result["evaluator_model"] == "mock-model"

    def test_latency_ms_is_positive(self):
        with _mock_evaluator():
            with patch("ragas.metrics.collections.AnswerRelevancy") as MockAR:
                MockAR.return_value.score.return_value = _make_metric_result(0.8)
                result = score_response("Q?", "A.", contexts=[])

        assert result["latency_ms"] >= 0


# ── _build_evaluator (async-compatibility checks) ────────────────────────────

class TestBuildEvaluator:
    """
    Instantiates real RAGAS objects with a fake API key — no HTTP calls at init.
    Checks .is_async on the returned objects, which is the exact flag that gates
    agenerate() and aembed_text(). Mocking the RAGAS classes entirely hides this
    seam: both async-client bugs would have been caught immediately by these tests.
    """

    def test_llm_is_async(self):
        """llm.is_async must be True; False raises 'Cannot use agenerate()' at eval time."""
        with patch("core.eval_runner.settings", openai_api_key="sk-test"):
            llm, _, _ = _build_evaluator()
        assert llm.is_async

    def test_embeddings_is_async(self):
        """emb.is_async must be True; False raises 'Cannot use aembed_text()' at eval time."""
        with patch("core.eval_runner.settings", openai_api_key="sk-test"):
            _, emb, _ = _build_evaluator()
        assert emb.is_async

    def test_returns_model_name_string(self):
        with patch("core.eval_runner.settings", openai_api_key="sk-test"):
            _, _, model_name = _build_evaluator()
        assert isinstance(model_name, str) and model_name != ""


# ── score_one_row ─────────────────────────────────────────────────────────────

def _fake_llm_emb():
    """Pre-built fake llm/embeddings matching the shape score_one_row expects."""
    return MagicMock(), MagicMock(), "mock-model"


class TestScoreOneRow:
    def _mock_metrics(self, ar=0.9, fa=0.8, cp=0.7, cr=0.6, ac=0.75):
        """Patch all five RAGAS metric classes in the collections module."""
        def _mr(v):
            m = MagicMock()
            m.value = v
            return m

        patches = {
            "ragas.metrics.collections.AnswerRelevancy": MagicMock(
                return_value=MagicMock(score=MagicMock(return_value=_mr(ar)))
            ),
            "ragas.metrics.collections.Faithfulness": MagicMock(
                return_value=MagicMock(score=MagicMock(return_value=_mr(fa)))
            ),
            "ragas.metrics.collections.ContextPrecisionWithoutReference": MagicMock(
                return_value=MagicMock(score=MagicMock(return_value=_mr(cp)))
            ),
            "ragas.metrics.collections.ContextRecall": MagicMock(
                return_value=MagicMock(score=MagicMock(return_value=_mr(cr)))
            ),
            "ragas.metrics.collections.AnswerCorrectness": MagicMock(
                return_value=MagicMock(score=MagicMock(return_value=_mr(ac)))
            ),
        }
        return patches

    def test_returns_all_keys(self):
        llm, emb, model = _fake_llm_emb()
        with patch.multiple("ragas.metrics.collections", **{
            k.split(".")[-1]: v for k, v in self._mock_metrics().items()
        }):
            result = score_one_row("Q?", "A.", [], None, llm, emb, model)
        assert set(result.keys()) == {
            "question", "answer", "ground_truth",
            "answer_relevancy", "faithfulness", "context_precision",
            "context_recall", "answer_correctness",
            "evaluator_model", "latency_ms", "error",
        }

    def test_no_contexts_skips_rag_metrics(self):
        llm, emb, model = _fake_llm_emb()
        with patch.multiple("ragas.metrics.collections", **{
            k.split(".")[-1]: v for k, v in self._mock_metrics().items()
        }):
            result = score_one_row("Q?", "A.", [], None, llm, emb, model)
        assert result["faithfulness"] is None
        assert result["context_precision"] is None
        assert result["context_recall"] is None

    def test_no_ground_truth_skips_gt_metrics(self):
        llm, emb, model = _fake_llm_emb()
        with patch.multiple("ragas.metrics.collections", **{
            k.split(".")[-1]: v for k, v in self._mock_metrics().items()
        }):
            result = score_one_row("Q?", "A.", ["chunk"], None, llm, emb, model)
        assert result["context_recall"] is None
        assert result["answer_correctness"] is None

    def test_all_five_metrics_with_gt_and_contexts(self):
        llm, emb, model = _fake_llm_emb()
        with patch.multiple("ragas.metrics.collections", **{
            k.split(".")[-1]: v for k, v in self._mock_metrics(
                ar=0.9, fa=0.8, cp=0.7, cr=0.6, ac=0.75
            ).items()
        }):
            result = score_one_row("Q?", "A.", ["chunk"], "ground truth", llm, emb, model)
        assert result["answer_relevancy"] == 0.9
        assert result["faithfulness"] == 0.8
        assert result["context_precision"] == 0.7
        assert result["context_recall"] == 0.6
        assert result["answer_correctness"] == 0.75

    def test_error_captured_not_raised(self):
        llm, emb, model = _fake_llm_emb()
        bad_metric = MagicMock(return_value=MagicMock(
            score=MagicMock(side_effect=RuntimeError("boom"))
        ))
        with patch("ragas.metrics.collections.AnswerRelevancy", bad_metric):
            result = score_one_row("Q?", "A.", [], None, llm, emb, model)
        assert result["error"] == "boom"
        assert result["answer_relevancy"] is None

    def test_evaluator_model_propagated(self):
        llm, emb, _ = _fake_llm_emb()
        with patch.multiple("ragas.metrics.collections", **{
            k.split(".")[-1]: v for k, v in self._mock_metrics().items()
        }):
            result = score_one_row("Q?", "A.", [], None, llm, emb, "my-model")
        assert result["evaluator_model"] == "my-model"

    def test_answer_correctness_receives_embeddings(self):
        """AnswerCorrectness requires embeddings for semantic similarity (default weights=[0.75,0.25]).
        Omitting embeddings raises ValueError at construction time — this test catches that."""
        llm, emb, model = _fake_llm_emb()
        MockAC = MagicMock()
        MockAC.return_value.score.return_value = _make_metric_result(0.8)

        with patch.multiple("ragas.metrics.collections", **{
            k.split(".")[-1]: v for k, v in self._mock_metrics().items()
        }), patch("ragas.metrics.collections.AnswerCorrectness", MockAC):
            score_one_row("Q?", "A.", ["ctx"], "ground truth", llm, emb, model)

        _, kwargs = MockAC.call_args
        assert kwargs.get("embeddings") is emb, (
            "AnswerCorrectness must receive embeddings; omitting them raises "
            "'embeddings are required for semantic similarity scoring'"
        )


# ── _parse_eval_csv ───────────────────────────────────────────────────────────

class TestParseEvalCsv:
    def _write_csv(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        f.write(content)
        f.close()
        return f.name

    def teardown_method(self):
        # tempfiles are cleaned up by the OS; explicit cleanup not required
        pass

    def test_question_only_csv(self):
        from ui.gradio_app import _parse_eval_csv
        path = self._write_csv("question\nWhat is AI?\nWhat is RAG?\n")
        rows = _parse_eval_csv(path)
        os.unlink(path)
        assert len(rows) == 2
        assert rows[0] == {"question": "What is AI?", "ground_truth": None}

    def test_question_and_ground_truth(self):
        from ui.gradio_app import _parse_eval_csv
        path = self._write_csv("question,ground_truth\nWhat is AI?,AI is...\n")
        rows = _parse_eval_csv(path)
        os.unlink(path)
        assert rows[0]["ground_truth"] == "AI is..."

    def test_missing_question_column_raises(self):
        from ui.gradio_app import _parse_eval_csv
        path = self._write_csv("query,answer\nfoo,bar\n")
        with pytest.raises(ValueError, match="question"):
            _parse_eval_csv(path)
        os.unlink(path)

    def test_empty_rows_skipped(self):
        from ui.gradio_app import _parse_eval_csv
        path = self._write_csv("question\nValid question\n\n  \n")
        rows = _parse_eval_csv(path)
        os.unlink(path)
        assert len(rows) == 1


# ── integration (skipped by default) ─────────────────────────────────────────

@pytest.fixture
def run_integration(request):
    return request.config.getoption("--run-integration")


@pytest.mark.integration
def test_real_api_end_to_end(run_integration):
    """Calls OpenAI + RAGAS for real. Run with: pytest --run-integration"""
    if not run_integration:
        pytest.skip("Pass --run-integration to run this test")

    result = score_response(
        question="What is the capital of France?",
        answer="The capital of France is Paris.",
        contexts=[],
    )
    assert result["error"] is None, f"Eval failed: {result['error']}"
    assert 0.0 <= result["answer_relevancy"] <= 1.0
    assert result["evaluator_model"] != ""
    assert result["latency_ms"] > 0
