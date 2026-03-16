"""
End-to-end integration tests for GoodMem client.

These tests run against a live GoodMem API server and verify all core
operations: create space, create memory (text and PDF), retrieve memories,
get memory, and delete memory.

Configuration is read from environment variables:
    GOODMEM_BASE_URL  - GoodMem API server URL (default: http://localhost:8080)
    GOODMEM_API_KEY   - GoodMem API key
    GOODMEM_PDF_PATH  - Path to a PDF file for testing PDF memory creation
"""

import os
import time
import uuid

import pytest

from deepeval.integrations.goodmem import (
    GoodMemClient,
    GoodMemEmbedder,
    GoodMemMemory,
    GoodMemRetrievalResult,
    GoodMemSpace,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("GOODMEM_BASE_URL", "http://localhost:8080")
API_KEY = os.environ.get("GOODMEM_API_KEY", "")
PDF_PATH = os.environ.get("GOODMEM_PDF_PATH", "")
EMBEDDER_ID = os.environ.get("GOODMEM_EMBEDDER_ID", "")


@pytest.fixture(scope="module")
def client() -> GoodMemClient:
    assert API_KEY, "GOODMEM_API_KEY environment variable must be set"
    return GoodMemClient(base_url=BASE_URL, api_key=API_KEY)


@pytest.fixture(scope="module")
def embedder_id(client: GoodMemClient) -> str:
    """Get the embedder ID from env or the first available one."""
    if EMBEDDER_ID:
        return EMBEDDER_ID
    embedders = client.list_embedders()
    assert len(embedders) > 0, "No embedders available on the GoodMem server"
    return embedders[0].embedder_id


@pytest.fixture(scope="module")
def space(client: GoodMemClient, embedder_id: str) -> GoodMemSpace:
    """Create a test space for the module."""
    unique_name = f"deepeval-test-{uuid.uuid4().hex[:8]}"
    space = client.create_space(name=unique_name, embedder_id=embedder_id)
    return space


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListEmbedders:
    def test_list_embedders(self, client: GoodMemClient) -> None:
        embedders = client.list_embedders()
        assert isinstance(embedders, list)
        assert len(embedders) > 0
        assert isinstance(embedders[0], GoodMemEmbedder)
        assert embedders[0].embedder_id
        print(f"  Found {len(embedders)} embedders: {[e.embedder_id for e in embedders]}")


class TestListSpaces:
    def test_list_spaces(self, client: GoodMemClient) -> None:
        spaces = client.list_spaces()
        assert isinstance(spaces, list)
        print(f"  Found {len(spaces)} spaces")


class TestCreateSpace:
    def test_create_space(self, space: GoodMemSpace) -> None:
        assert space.space_id
        assert space.name
        print(f"  Created space: id={space.space_id}, name={space.name}, reused={space.reused}")

    def test_create_space_reuse(
        self, client: GoodMemClient, space: GoodMemSpace, embedder_id: str
    ) -> None:
        """Creating a space with the same name should reuse the existing one."""
        reused = client.create_space(name=space.name, embedder_id=embedder_id)
        assert reused.space_id == space.space_id
        assert reused.reused is True
        print(f"  Reused space: id={reused.space_id}")


class TestCreateMemoryText:
    def test_create_memory_text(
        self, client: GoodMemClient, space: GoodMemSpace
    ) -> None:
        memory = client.create_memory(
            space_id=space.space_id,
            text_content="DeepEval is an open-source evaluation framework for LLMs. It supports RAG evaluation with metrics like faithfulness, contextual relevancy, and answer relevancy.",
        )
        assert isinstance(memory, GoodMemMemory)
        assert memory.memory_id
        assert memory.space_id == space.space_id
        assert memory.content_type == "text/plain"
        print(f"  Created text memory: id={memory.memory_id}, status={memory.processing_status}")


class TestCreateMemoryPDF:
    def test_create_memory_pdf(
        self, client: GoodMemClient, space: GoodMemSpace
    ) -> None:
        if not PDF_PATH or not os.path.exists(PDF_PATH):
            pytest.skip(f"PDF file not found at: {PDF_PATH}")

        memory = client.create_memory(
            space_id=space.space_id,
            file_path=PDF_PATH,
        )
        assert isinstance(memory, GoodMemMemory)
        assert memory.memory_id
        assert memory.content_type == "application/pdf"
        assert memory.file_name is not None
        print(f"  Created PDF memory: id={memory.memory_id}, file={memory.file_name}, status={memory.processing_status}")


class TestRetrieveMemories:
    def test_retrieve_memories(
        self, client: GoodMemClient, space: GoodMemSpace
    ) -> None:
        # First create a memory to ensure there's something to retrieve
        client.create_memory(
            space_id=space.space_id,
            text_content="Python is a programming language known for its simplicity and readability. It is widely used in machine learning and data science.",
        )

        # Retrieve with wait_for_indexing=True to handle processing delay
        result = client.retrieve_memories(
            query="What programming language is used for machine learning?",
            space_ids=space.space_id,
            max_results=5,
            wait_for_indexing=True,
        )
        assert isinstance(result, GoodMemRetrievalResult)
        assert result.total_results > 0
        assert result.results[0].chunk_text is not None
        print(f"  Retrieved {result.total_results} results")
        for i, chunk in enumerate(result.results[:3]):
            print(f"    [{i}] score={chunk.relevance_score}, text={chunk.chunk_text[:80]}...")

    def test_retrieve_as_context(
        self, client: GoodMemClient, space: GoodMemSpace
    ) -> None:
        """Test the convenience method for deepeval integration."""
        contexts = client.retrieve_as_context(
            query="evaluation framework",
            space_ids=space.space_id,
            max_results=3,
            wait_for_indexing=True,
        )
        assert isinstance(contexts, list)
        assert len(contexts) > 0
        assert all(isinstance(c, str) for c in contexts)
        print(f"  retrieve_as_context returned {len(contexts)} context strings")


class TestGetMemory:
    def test_get_memory(
        self, client: GoodMemClient, space: GoodMemSpace
    ) -> None:
        # Create a memory to get
        memory = client.create_memory(
            space_id=space.space_id,
            text_content="Test content for get_memory verification.",
        )
        result = client.get_memory(memory.memory_id, include_content=True)
        assert "memory" in result
        assert result["memory"].get("memoryId") == memory.memory_id
        print(f"  Got memory: id={memory.memory_id}, keys={list(result.keys())}")


class TestDeleteMemory:
    def test_delete_memory(
        self, client: GoodMemClient, space: GoodMemSpace
    ) -> None:
        # Create a memory to delete
        memory = client.create_memory(
            space_id=space.space_id,
            text_content="Temporary content that will be deleted.",
        )
        result = client.delete_memory(memory.memory_id)
        assert result["success"] is True
        assert result["memory_id"] == memory.memory_id
        print(f"  Deleted memory: id={memory.memory_id}")

        # Verify it's actually gone
        with pytest.raises(Exception):
            client.get_memory(memory.memory_id)
        print("  Verified memory is no longer accessible")
