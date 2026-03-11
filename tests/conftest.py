from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from goodmem_deepeval.client import GoodMemEvalClient
from goodmem_deepeval.retriever import GoodMemChunk, GoodMemRetriever


def _make_retrieval_result(
    content: str = "chunk text",
    memory_id: str = "mem-1",
    space_id: str = "space-1",
    score: float = 0.9,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "memory_id": memory_id,
        "space_id": space_id,
        "score": score,
        "content": content,
        "metadata": metadata or {},
    }


@pytest.fixture()
def sample_retrieval() -> Dict[str, Any]:
    return {
        "query": "test query",
        "results": [
            _make_retrieval_result("first chunk", "mem-1", "sp-1", 0.95, {"source": "docs"}),
            _make_retrieval_result("second chunk", "mem-2", "sp-1", 0.80),
        ],
    }


@pytest.fixture()
def sample_chunks() -> List[GoodMemChunk]:
    return [
        GoodMemChunk(content="first chunk", memory_id="mem-1", space_id="sp-1", score=0.95, metadata={"source": "docs"}),
        GoodMemChunk(content="second chunk", memory_id="mem-2", space_id="sp-1", score=0.80, metadata={}),
    ]


@pytest.fixture()
def mock_client(sample_retrieval: Dict[str, Any]) -> GoodMemEvalClient:
    """GoodMemEvalClient with all SDK calls mocked out."""
    client = MagicMock(spec=GoodMemEvalClient)
    client.retrieve_memories.return_value = sample_retrieval
    client.create_space.return_value = MagicMock(space_id="space-1", name="test-space")
    client.create_memory.return_value = MagicMock(memory_id="mem-1")
    client.list_spaces.return_value = []
    client.list_memories.return_value = []
    client.get_memory.return_value = MagicMock(memory_id="mem-1")
    client.delete_memory.return_value = None
    return client


@pytest.fixture()
def mock_retriever(mock_client: GoodMemEvalClient) -> GoodMemRetriever:
    return GoodMemRetriever(client=mock_client, spaces=["space-1"], maximum_results=5)
