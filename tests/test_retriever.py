from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

from goodmem_deepeval.retriever import GoodMemChunk, GoodMemRetriever
from goodmem_deepeval.client import GoodMemEvalClient


class TestGoodMemChunk:
    def test_source_property(self):
        chunk = GoodMemChunk(content="x", memory_id=None, space_id=None, score=None, metadata={"source": "docs"})
        assert chunk.source == "docs"

    def test_author_property(self):
        chunk = GoodMemChunk(content="x", memory_id=None, space_id=None, score=None, metadata={"author": "bob"})
        assert chunk.author == "bob"

    def test_tags_from_list(self):
        chunk = GoodMemChunk(content="x", memory_id=None, space_id=None, score=None, metadata={"tags": ["a", "b"]})
        assert list(chunk.tags) == ["a", "b"]

    def test_tags_from_csv_string(self):
        chunk = GoodMemChunk(content="x", memory_id=None, space_id=None, score=None, metadata={"tags": "a, b, c"})
        assert list(chunk.tags) == ["a", "b", "c"]

    def test_tags_empty(self):
        chunk = GoodMemChunk(content="x", memory_id=None, space_id=None, score=None, metadata={})
        assert list(chunk.tags) == []


class TestGoodMemRetriever:
    def test_retrieve_returns_chunks(self, mock_client: GoodMemEvalClient, sample_retrieval: Dict[str, Any]):
        retriever = GoodMemRetriever(client=mock_client, spaces=["sp-1"])
        chunks = retriever.retrieve("test")

        assert len(chunks) == 2
        assert chunks[0].content == "first chunk"
        assert chunks[0].memory_id == "mem-1"
        assert chunks[1].content == "second chunk"

    def test_forwards_reranker(self, mock_client: GoodMemEvalClient):
        retriever = GoodMemRetriever(
            client=mock_client,
            spaces=["sp-1"],
            reranker="reranker-1",
            llm="llm-1",
            llm_temperature=0.5,
            chronological_resort=True,
        )
        retriever.retrieve("test")

        call_kwargs = mock_client.retrieve_memories.call_args.kwargs
        assert call_kwargs["reranker"] == "reranker-1"
        assert call_kwargs["llm"] == "llm-1"
        assert call_kwargs["llm_temperature"] == 0.5
        assert call_kwargs["chronological_resort"] is True

    def test_metadata_filter_wrapped(self, mock_client: GoodMemEvalClient):
        retriever = GoodMemRetriever(
            client=mock_client,
            spaces=["sp-1"],
            metadata_filter="CAST(val('$.source') AS TEXT) = 'docs'",
        )
        retriever.retrieve("test")

        call_kwargs = mock_client.retrieve_memories.call_args.kwargs
        assert call_kwargs["metadata_filters"] == {"filter": "CAST(val('$.source') AS TEXT) = 'docs'"}

    def test_no_metadata_filter(self, mock_client: GoodMemEvalClient):
        retriever = GoodMemRetriever(client=mock_client, spaces=["sp-1"])
        retriever.retrieve("test")

        call_kwargs = mock_client.retrieve_memories.call_args.kwargs
        assert call_kwargs["metadata_filters"] is None

    def test_to_text_list(self, sample_chunks: List[GoodMemChunk]):
        result = GoodMemRetriever.to_text_list(sample_chunks)
        assert result == ["first chunk", "second chunk"]
