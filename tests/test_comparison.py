from __future__ import annotations

from typing import List, Tuple
from unittest.mock import patch, MagicMock

from goodmem_deepeval.comparison import compare_pipelines


class TestComparePipelines:
    def test_returns_results_per_system(self):
        def pipeline_a(query: str) -> Tuple[str, List[str]]:
            return "answer_a", ["ctx_a"]

        def pipeline_b(query: str) -> Tuple[str, List[str]]:
            return "answer_b", ["ctx_b"]

        mock_eval_result = MagicMock()

        with patch("goodmem_deepeval.comparison.evaluate", return_value=mock_eval_result) as mock_eval:
            results = compare_pipelines(
                queries=["q1", "q2"],
                pipelines={"system_a": pipeline_a, "system_b": pipeline_b},
                metrics=[MagicMock()],
            )

        assert "system_a" in results
        assert "system_b" in results
        assert mock_eval.call_count == 2

    def test_single_pipeline(self):
        def pipeline(query: str) -> Tuple[str, List[str]]:
            return "ans", ["ctx"]

        with patch("goodmem_deepeval.comparison.evaluate", return_value=MagicMock()):
            results = compare_pipelines(
                queries=["q"],
                pipelines={"only": pipeline},
                metrics=[],
            )

        assert len(results) == 1
        assert "only" in results
