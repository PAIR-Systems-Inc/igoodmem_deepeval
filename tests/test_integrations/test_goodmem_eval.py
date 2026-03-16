#!/usr/bin/env python3
"""
DeepEval evaluation tests for GoodMem integration.

Runs deepeval metrics (AnswerRelevancy, Faithfulness, ContextualRelevancy)
against a live GoodMem server using retrieve_as_context to populate
LLMTestCase.retrieval_context.

Usage:
    GOODMEM_API_KEY="..." OPENAI_API_KEY="..." python3 -m pytest tests/test_integrations/test_goodmem_eval.py -v -s
"""

import os
import time
import uuid

import pytest

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.integrations.goodmem import GoodMemClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("GOODMEM_BASE_URL", "http://localhost:8080")
API_KEY = os.environ.get("GOODMEM_API_KEY", "")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> GoodMemClient:
    assert API_KEY, "GOODMEM_API_KEY environment variable must be set"
    return GoodMemClient(base_url=BASE_URL, api_key=API_KEY)


@pytest.fixture(scope="module")
def space_with_data(client: GoodMemClient):
    """Create a space, populate it with text memories, and return the space."""
    # Prefer the Voyage embedder for faster/more reliable indexing.
    # Fall back to the first available embedder if Voyage is not found.
    VOYAGE_EMBEDDER_ID = os.environ.get("GOODMEM_EMBEDDER_ID", "")
    embedders = client.list_embedders()
    assert len(embedders) > 0, "No embedders available on the GoodMem server"
    if VOYAGE_EMBEDDER_ID:
        embedder_id = VOYAGE_EMBEDDER_ID
    else:
        # Try to find a Voyage embedder by name
        voyage = [e for e in embedders if "voyage" in (e.display_name or "").lower()]
        embedder_id = voyage[0].embedder_id if voyage else embedders[0].embedder_id
    print(f"\n  Using embedder: {embedder_id}")

    unique_name = f"deepeval-eval-{uuid.uuid4().hex[:8]}"
    space = client.create_space(name=unique_name, embedder_id=embedder_id)

    # Seed knowledge into the space
    documents = [
        (
            "DeepEval is an open-source evaluation framework for LLMs. "
            "It provides metrics like faithfulness, contextual relevancy, "
            "answer relevancy, bias, and toxicity to evaluate LLM outputs. "
            "DeepEval supports RAG evaluation and agentic workflows."
        ),
        (
            "Python is a high-level programming language known for its "
            "simplicity and readability. It is widely used in machine learning, "
            "data science, web development, and automation. Popular Python ML "
            "frameworks include TensorFlow, PyTorch, and scikit-learn."
        ),
        (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "information retrieval with text generation. RAG systems retrieve "
            "relevant documents from a knowledge base and use them as context "
            "for generating accurate, grounded responses."
        ),
    ]

    for doc in documents:
        client.create_memory(space_id=space.space_id, text_content=doc)

    # Wait for indexing to complete before running any retrieval tests.
    # The server needs time to chunk and embed newly created memories.
    print("\n  Waiting for GoodMem indexing (polling up to 120s)...")
    deadline = time.time() + 120
    while time.time() < deadline:
        result = client.retrieve_memories(
            query="DeepEval",
            space_ids=space.space_id,
            max_results=1,
            wait_for_indexing=False,
        )
        if result.total_results > 0:
            print(f"  Indexing ready ({result.total_results} result(s) found).")
            break
        time.sleep(5)
    else:
        print("  WARNING: Indexing may not be complete after 120s.")

    return space


# ---------------------------------------------------------------------------
# Test cases: (query, simulated_actual_output, expected_output)
# ---------------------------------------------------------------------------

EVAL_CASES = [
    {
        "input": "What is DeepEval and what metrics does it support?",
        "actual_output": (
            "DeepEval is an open-source evaluation framework for LLMs. "
            "It supports metrics such as faithfulness, contextual relevancy, "
            "answer relevancy, bias, and toxicity."
        ),
        "expected_output": (
            "DeepEval is an open-source LLM evaluation framework that supports "
            "metrics like faithfulness, contextual relevancy, and answer relevancy."
        ),
    },
    {
        "input": "What is RAG and how does it work?",
        "actual_output": (
            "RAG stands for Retrieval-Augmented Generation. It retrieves "
            "relevant documents from a knowledge base and uses them as context "
            "to generate accurate responses."
        ),
        "expected_output": (
            "RAG combines information retrieval with text generation to produce "
            "grounded, accurate answers."
        ),
    },
    {
        "input": "What programming language is popular for machine learning?",
        "actual_output": (
            "Python is the most popular programming language for machine learning. "
            "It supports frameworks like TensorFlow, PyTorch, and scikit-learn."
        ),
        "expected_output": (
            "Python is widely used for machine learning with frameworks like "
            "TensorFlow and PyTorch."
        ),
    },
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

answer_relevancy = AnswerRelevancyMetric(threshold=0.5)
faithfulness = FaithfulnessMetric(threshold=0.5)
contextual_relevancy = ContextualRelevancyMetric(threshold=0.5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["input"][:50] for c in EVAL_CASES])
def test_goodmem_eval(client, space_with_data, case):
    """Evaluate GoodMem retrieval with deepeval metrics."""
    # Retrieve context from GoodMem (with extended retry for slow indexing)
    retrieval_context = []
    deadline = time.time() + 90
    while time.time() < deadline:
        retrieval_context = client.retrieve_as_context(
            query=case["input"],
            space_ids=space_with_data.space_id,
            max_results=5,
            wait_for_indexing=False,
        )
        if retrieval_context:
            break
        time.sleep(5)

    print(f"\n  Retrieved {len(retrieval_context)} context chunks for: {case['input'][:60]}")
    for i, ctx in enumerate(retrieval_context[:3]):
        print(f"    [{i}] {ctx[:100]}...")

    assert len(retrieval_context) > 0, "No retrieval context returned from GoodMem"

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        expected_output=case["expected_output"],
        retrieval_context=retrieval_context,
    )

    # Run all metrics
    evaluate(
        test_cases=[test_case],
        metrics=[answer_relevancy, faithfulness, contextual_relevancy],
        display_config=DisplayConfig(print_results=True, verbose_mode=True),
        async_config=AsyncConfig(run_async=False),
    )
