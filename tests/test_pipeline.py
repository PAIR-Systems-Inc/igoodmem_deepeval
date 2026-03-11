from __future__ import annotations

from goodmem_deepeval.pipeline import GoodMemRAGPipeline
from goodmem_deepeval.retriever import GoodMemRetriever
from goodmem_deepeval.client import GoodMemEvalClient


class TestGoodMemRAGPipeline:
    def test_call_returns_answer_and_context(self, mock_client: GoodMemEvalClient):
        retriever = GoodMemRetriever(client=mock_client, spaces=["sp-1"])

        def generator(query: str, context: list[str]) -> str:
            return f"answer: {context[0]}" if context else "empty"

        pipeline = GoodMemRAGPipeline(retriever=retriever, generator=generator)
        answer, ctx = pipeline("test query")

        assert answer == "answer: first chunk"
        assert ctx == ["first chunk", "second chunk"]

    def test_empty_retrieval(self, mock_client: GoodMemEvalClient):
        mock_client.retrieve_memories.return_value = {"query": "q", "results": []}
        retriever = GoodMemRetriever(client=mock_client, spaces=["sp-1"])

        def generator(query: str, context: list[str]) -> str:
            return "no context"

        pipeline = GoodMemRAGPipeline(retriever=retriever, generator=generator)
        answer, ctx = pipeline("test")

        assert answer == "no context"
        assert ctx == []
