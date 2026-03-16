#!/usr/bin/env python3
"""
Standalone end-to-end integration tests for GoodMem client.

Runs against a live GoodMem API server without requiring the full deepeval
dependency tree. Uses importlib to load modules directly.

Usage:
    GOODMEM_API_KEY="..." GOODMEM_PDF_PATH="..." python3 tests/test_integrations/test_goodmem_e2e.py
"""

import importlib.util
import os
import sys
import time
import traceback
import uuid


def _load_module(name, path):
    """Load a Python module from a file path without triggering package __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load types first (it has no deepeval imports), then client
_repo = os.path.join(os.path.dirname(__file__), "..", "..")
_types_mod = _load_module(
    "deepeval.integrations.goodmem.types",
    os.path.join(_repo, "deepeval", "integrations", "goodmem", "types.py"),
)
# Patch sys.modules so client.py can import types
sys.modules["deepeval.integrations.goodmem.types"] = _types_mod

_client_mod = _load_module(
    "deepeval.integrations.goodmem.client",
    os.path.join(_repo, "deepeval", "integrations", "goodmem", "client.py"),
)

GoodMemClient = _client_mod.GoodMemClient
GoodMemEmbedder = _types_mod.GoodMemEmbedder
GoodMemMemory = _types_mod.GoodMemMemory
GoodMemRetrievalResult = _types_mod.GoodMemRetrievalResult
GoodMemSpace = _types_mod.GoodMemSpace

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("GOODMEM_BASE_URL", "http://localhost:8080")
API_KEY = os.environ.get("GOODMEM_API_KEY", "")
PDF_PATH = os.environ.get("GOODMEM_PDF_PATH", "")
# Optional: specify a known-working embedder ID to avoid embedder failures
EMBEDDER_ID = os.environ.get("GOODMEM_EMBEDDER_ID", "")

RESULTS = []


def record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    RESULTS.append({"name": name, "status": status, "detail": detail})
    print(f"  [{status}] {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_embedders(client):
    """Test listing embedders and return the first embedder_id."""
    embedders = client.list_embedders()
    assert len(embedders) > 0, "No embedders available"
    assert isinstance(embedders[0], GoodMemEmbedder)
    assert embedders[0].embedder_id
    record(
        "list_embedders",
        True,
        f"Found {len(embedders)} embedders: {[e.embedder_id for e in embedders]}",
    )
    return embedders[0].embedder_id


def test_list_spaces(client):
    """Test listing spaces."""
    spaces = client.list_spaces()
    assert isinstance(spaces, list)
    record("list_spaces", True, f"Found {len(spaces)} spaces")


def test_create_space(client, embedder_id):
    """Test creating a new space."""
    unique_name = f"deepeval-test-{uuid.uuid4().hex[:8]}"
    space = client.create_space(name=unique_name, embedder_id=embedder_id)
    assert isinstance(space, GoodMemSpace)
    assert space.space_id
    assert space.name == unique_name
    record(
        "create_space",
        True,
        f"id={space.space_id}, name={space.name}, reused={space.reused}",
    )
    return space


def test_create_space_reuse(client, space, embedder_id):
    """Test that creating a space with the same name reuses the existing one."""
    reused = client.create_space(name=space.name, embedder_id=embedder_id)
    assert reused.space_id == space.space_id
    assert reused.reused is True
    record("create_space_reuse", True, f"Reused space: id={reused.space_id}")


def test_create_memory_text(client, space):
    """Test creating a memory with plain text content."""
    memory = client.create_memory(
        space_id=space.space_id,
        text_content=(
            "DeepEval is an open-source evaluation framework for LLMs. "
            "It supports RAG evaluation with metrics like faithfulness, "
            "contextual relevancy, and answer relevancy. DeepEval helps "
            "developers test their LLM applications systematically."
        ),
    )
    assert isinstance(memory, GoodMemMemory)
    assert memory.memory_id
    assert memory.space_id == space.space_id
    assert memory.content_type == "text/plain"
    record(
        "create_memory_text",
        True,
        f"id={memory.memory_id}, status={memory.processing_status}",
    )
    return memory


def test_create_memory_pdf(client, space):
    """Test creating a memory from a PDF file."""
    if not PDF_PATH or not os.path.exists(PDF_PATH):
        record("create_memory_pdf", False, f"PDF file not found at: {PDF_PATH}")
        return None

    memory = client.create_memory(
        space_id=space.space_id,
        file_path=PDF_PATH,
    )
    assert isinstance(memory, GoodMemMemory)
    assert memory.memory_id
    assert memory.content_type == "application/pdf"
    assert memory.file_name is not None
    record(
        "create_memory_pdf",
        True,
        f"id={memory.memory_id}, file={memory.file_name}, status={memory.processing_status}",
    )
    return memory


def test_retrieve_memories(client, space):
    """Test retrieving memories via semantic search."""
    result = client.retrieve_memories(
        query="What is DeepEval and how does it evaluate LLMs?",
        space_ids=space.space_id,
        max_results=5,
        wait_for_indexing=True,
    )
    assert isinstance(result, GoodMemRetrievalResult)
    assert result.total_results > 0, "No results returned (indexing may have timed out)"
    assert result.results[0].chunk_text is not None
    detail_lines = [f"Retrieved {result.total_results} results"]
    for i, chunk in enumerate(result.results[:3]):
        text_preview = (chunk.chunk_text or "")[:100]
        detail_lines.append(f"  [{i}] score={chunk.relevance_score}, text={text_preview}...")
    record("retrieve_memories", True, "\n".join(detail_lines))


def test_retrieve_as_context(client, space):
    """Test the convenience method for deepeval LLMTestCase integration."""
    contexts = client.retrieve_as_context(
        query="evaluation framework for LLMs",
        space_ids=space.space_id,
        max_results=3,
        wait_for_indexing=True,
    )
    assert isinstance(contexts, list)
    assert len(contexts) > 0
    assert all(isinstance(c, str) for c in contexts)
    record(
        "retrieve_as_context",
        True,
        f"Returned {len(contexts)} context strings for LLMTestCase.retrieval_context",
    )


def test_get_memory(client, space):
    """Test fetching a specific memory by ID."""
    memory = client.create_memory(
        space_id=space.space_id,
        text_content="Test content for get_memory verification.",
    )
    result = client.get_memory(memory.memory_id, include_content=True)
    assert "memory" in result
    assert result["memory"].get("memoryId") == memory.memory_id
    record(
        "get_memory",
        True,
        f"id={memory.memory_id}, keys={list(result.keys())}",
    )


def test_delete_memory(client, space):
    """Test deleting a memory."""
    memory = client.create_memory(
        space_id=space.space_id,
        text_content="Temporary content that will be deleted.",
    )
    result = client.delete_memory(memory.memory_id)
    assert result["success"] is True
    assert result["memory_id"] == memory.memory_id

    # Verify it's actually gone
    try:
        client.get_memory(memory.memory_id)
        record("delete_memory", False, "Memory was not actually deleted")
    except Exception:
        record(
            "delete_memory",
            True,
            f"Deleted id={memory.memory_id}, confirmed inaccessible",
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    if not API_KEY:
        print("ERROR: GOODMEM_API_KEY environment variable must be set")
        sys.exit(1)

    print(f"GoodMem E2E Tests")
    print(f"  Base URL: {BASE_URL}")
    print(f"  API Key:  {API_KEY[:10]}...")
    print(f"  PDF Path: {PDF_PATH or '(not set)'}")
    print()

    client = GoodMemClient(base_url=BASE_URL, api_key=API_KEY)

    # Phase 1: List operations
    embedder_id = EMBEDDER_ID or None
    try:
        first_embedder = test_list_embedders(client)
        if not embedder_id:
            embedder_id = first_embedder
    except Exception as e:
        record("list_embedders", False, f"{e}\n{traceback.format_exc()}")

    try:
        test_list_spaces(client)
    except Exception as e:
        record("list_spaces", False, f"{e}\n{traceback.format_exc()}")

    if not embedder_id:
        print("\nFATAL: Could not get embedder_id, cannot continue")
        sys.exit(1)

    # Phase 2: Create space
    space = None
    try:
        space = test_create_space(client, embedder_id)
    except Exception as e:
        record("create_space", False, f"{e}\n{traceback.format_exc()}")
        print("\nFATAL: Could not create space, cannot continue")
        sys.exit(1)

    # Phase 3: Run remaining tests sequentially
    remaining_tests = [
        ("create_space_reuse", lambda: test_create_space_reuse(client, space, embedder_id)),
        ("create_memory_text", lambda: test_create_memory_text(client, space)),
        ("create_memory_pdf", lambda: test_create_memory_pdf(client, space)),
        ("retrieve_memories", lambda: test_retrieve_memories(client, space)),
        ("retrieve_as_context", lambda: test_retrieve_as_context(client, space)),
        ("get_memory", lambda: test_get_memory(client, space)),
        ("delete_memory", lambda: test_delete_memory(client, space)),
    ]

    for test_name, test_fn in remaining_tests:
        try:
            test_fn()
        except Exception as e:
            record(test_name, False, f"{e}\n{traceback.format_exc()}")

    # Summary
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    for r in RESULTS:
        print(f"  [{r['status']}] {r['name']}")
    print(f"\n  Total: {len(RESULTS)} | Passed: {passed} | Failed: {failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
