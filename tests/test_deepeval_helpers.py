from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

from deepeval.test_case import LLMTestCase

from goodmem_deepeval.deepeval_helpers import (
    build_llm_test_case_from_goodmem,
    build_llm_test_case_from_pipeline,
)
from goodmem_deepeval.pipeline import GoodMemRAGPipeline
from goodmem_deepeval.retriever import GoodMemRetriever
from goodmem_deepeval.client import GoodMemEvalClient


class TestBuildLLMTestCaseFromGoodmem:
    def test_basic(self, sample_retrieval: Dict[str, Any]):
        tc = build_llm_test_case_from_goodmem(
            query="q",
            actual_output="answer",
            retrieval=sample_retrieval,
            expected_output="expected",
        )
        assert isinstance(tc, LLMTestCase)
        assert tc.input == "q"
        assert tc.actual_output == "answer"
        assert tc.expected_output == "expected"
        assert tc.retrieval_context == ["first chunk", "second chunk"]

    def test_no_actual_output(self, sample_retrieval: Dict[str, Any]):
        tc = build_llm_test_case_from_goodmem(query="q", actual_output=None, retrieval=sample_retrieval)
        assert tc.actual_output == ""

    def test_empty_results(self):
        tc = build_llm_test_case_from_goodmem(
            query="q", actual_output="a", retrieval={"query": "q", "results": []}
        )
        assert tc.retrieval_context == []


class TestBuildLLMTestCaseFromPipeline:
    def test_uses_pipeline_output(self, mock_client: GoodMemEvalClient):
        retriever = GoodMemRetriever(client=mock_client, spaces=["sp-1"])

        def gen(q: str, ctx: list[str]) -> str:
            return "generated"

        pipeline = GoodMemRAGPipeline(retriever=retriever, generator=gen)
        tc = build_llm_test_case_from_pipeline(query="q", pipeline=pipeline, expected_output="exp")

        assert tc.input == "q"
        assert tc.actual_output == "generated"
        assert tc.expected_output == "exp"
        assert len(tc.retrieval_context) == 2
