"""
Live benchmark of the igoodmem_deepeval integration.
Runs end-to-end RAG eval and retrieval-only eval against a real GoodMem server.
"""
from __future__ import annotations

import os
import time
from typing import List, Tuple

from goodmem_client.api.embedders_api import EmbeddersApi
from goodmem_client.api_client import ApiClient
from goodmem_client.configuration import Configuration

from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemRetriever,
    GoodMemRAGPipeline,
    evaluate_goodmem_rag,
    evaluate_goodmem_retriever_only,
    compare_pipelines,
    build_llm_test_case_from_goodmem,
)

BASE_URL = os.environ["GOODMEM_BASE_URL"]
API_KEY = os.environ["GOODMEM_API_KEY"]

BENCHMARK_DOCS = [
    {
        "text": "GoodMem is a semantic memory platform that stores and retrieves information using vector embeddings. It supports multiple embedding models and can rerank results for improved relevance.",
        "source": "docs",
        "author": "engineering",
        "tags": "architecture,embeddings,reranking",
    },
    {
        "text": "Retrieval-Augmented Generation (RAG) combines a retriever component with a language model. The retriever fetches relevant context from a knowledge base, and the LLM generates answers grounded in that context.",
        "source": "docs",
        "author": "engineering",
        "tags": "rag,retrieval,llm",
    },
    {
        "text": "GoodMem supports metadata filtering on retrieval, allowing users to scope searches by source, author, tags, or custom fields. Filter expressions use SQL-like syntax on JSON metadata.",
        "source": "docs",
        "author": "engineering",
        "tags": "filtering,metadata,search",
    },
    {
        "text": "The marketing team reports that customer satisfaction improved by 35% after deploying GoodMem-powered support chatbots. Response accuracy increased significantly with semantic retrieval.",
        "source": "marketing",
        "author": "marketing-team",
        "tags": "customers,chatbot,satisfaction",
    },
    {
        "text": "Vector databases store high-dimensional embeddings and support approximate nearest neighbor search. GoodMem abstracts this complexity behind a simple API for memory storage and retrieval.",
        "source": "docs",
        "author": "engineering",
        "tags": "vector-db,embeddings,api",
    },
]

EVAL_QUERIES = [
    "How does GoodMem improve retrieval quality?",
    "What is RAG and how does it work?",
    "How does metadata filtering work in GoodMem?",
]


def setup_space(client: GoodMemEvalClient) -> str:
    """Create or reuse benchmark space, ingest docs."""
    # Find an embedder
    cfg = Configuration()
    cfg.host = BASE_URL
    cfg.api_key = {"ApiKeyAuth": API_KEY}
    embedder_client = ApiClient(configuration=cfg)
    embedders = EmbeddersApi(embedder_client).list_embedders().embedders
    if not embedders:
        raise RuntimeError("No embedders configured in GoodMem.")
    embedder_id = embedders[0].embedder_id
    print(f"Using embedder: {embedder_id}")

    # Create or reuse space
    spaces = client.list_spaces()
    existing = next((s for s in spaces if s.name == "deepeval-benchmark"), None)
    if existing:
        space_id = existing.space_id
        print(f"Reusing space: {space_id}")
    else:
        space = client.create_space(
            space_name="deepeval-benchmark",
            embedder=embedder_id,
            chunk_size=512,
            chunk_overlap=50,
        )
        space_id = space.space_id
        print(f"Created space: {space_id}")

        # Ingest documents
        for doc in BENCHMARK_DOCS:
            client.create_memory(
                space=space_id,
                text_content=doc["text"],
                source=doc["source"],
                author=doc["author"],
                tags=doc["tags"],
            )
        print(f"Ingested {len(BENCHMARK_DOCS)} memories")
        # Brief pause for indexing
        time.sleep(3)

    return space_id


def run_retrieval_only_eval(client: GoodMemEvalClient, space_id: str) -> None:
    """Evaluate retrieval quality only."""
    from deepeval.metrics import ContextualRelevancyMetric

    print("\n" + "=" * 60)
    print("RETRIEVAL-ONLY EVALUATION")
    print("=" * 60)

    result = evaluate_goodmem_retriever_only(
        queries=EVAL_QUERIES,
        client=client,
        spaces=[space_id],
        metrics=[ContextualRelevancyMetric(threshold=0.5)],
        maximum_results=3,
    )
    print(result)


def run_rag_eval(client: GoodMemEvalClient, space_id: str) -> None:
    """Evaluate end-to-end RAG pipeline."""
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

    print("\n" + "=" * 60)
    print("END-TO-END RAG EVALUATION")
    print("=" * 60)

    retriever = GoodMemRetriever(client=client, spaces=[space_id], maximum_results=3)

    def generator(query: str, context: list[str]) -> str:
        if context:
            return f"Based on the retrieved information: {' '.join(context[:2])}"
        return "No relevant information found."

    pipeline = GoodMemRAGPipeline(retriever=retriever, generator=generator)

    result = evaluate_goodmem_rag(
        queries=EVAL_QUERIES,
        pipeline=pipeline,
        metrics=[
            AnswerRelevancyMetric(threshold=0.5),
            FaithfulnessMetric(threshold=0.5),
        ],
    )
    print(result)


def run_comparison(client: GoodMemEvalClient, space_id: str) -> None:
    """Compare GoodMem pipeline vs a dummy baseline."""
    from deepeval.metrics import AnswerRelevancyMetric

    print("\n" + "=" * 60)
    print("GOODMEM vs BASELINE COMPARISON")
    print("=" * 60)

    retriever = GoodMemRetriever(client=client, spaces=[space_id], maximum_results=3)

    def gm_generator(query: str, context: list[str]) -> str:
        if context:
            return f"Based on GoodMem retrieval: {context[0]}"
        return "No context."

    goodmem_pipeline = GoodMemRAGPipeline(retriever=retriever, generator=gm_generator)

    def baseline_pipeline(query: str) -> Tuple[str, List[str]]:
        return (
            "I don't have specific information about that topic.",
            ["Generic baseline context that may not be relevant."],
        )

    results = compare_pipelines(
        queries=EVAL_QUERIES,
        pipelines={
            "goodmem": goodmem_pipeline,
            "baseline": baseline_pipeline,
        },
        metrics=[AnswerRelevancyMetric(threshold=0.5)],
    )

    for name, res in results.items():
        print(f"\n--- {name} ---")
        print(res)


def run_direct_test_case_demo(client: GoodMemEvalClient, space_id: str) -> None:
    """Demo the target API from the GitHub issue."""
    from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric
    from deepeval.test_case import LLMTestCase

    print("\n" + "=" * 60)
    print("DIRECT TEST CASE BUILD (Target API Demo)")
    print("=" * 60)

    retrieval = client.retrieve_memories(
        query="How does GoodMem improve retrieval quality?",
        spaces=[space_id],
        maximum_results=5,
    )

    print(f"Retrieved {len(retrieval['results'])} chunks")
    for i, r in enumerate(retrieval["results"]):
        print(f"  [{i}] score={r['score']:.3f}  content={r['content'][:80]}...")

    test_case = LLMTestCase(
        input="How does GoodMem improve retrieval quality?",
        actual_output="GoodMem improves retrieval quality with semantic search and optional reranking.",
        expected_output="GoodMem uses semantic retrieval and can improve ranking quality through reranking.",
        retrieval_context=[item["content"] for item in retrieval["results"]],
    )

    result = evaluate(
        test_cases=[test_case],
        metrics=[
            AnswerRelevancyMetric(threshold=0.5),
            ContextualRelevancyMetric(threshold=0.5),
        ],
    )
    print(result)


def main() -> None:
    client = GoodMemEvalClient(base_url=BASE_URL, api_key=API_KEY)
    space_id = setup_space(client)

    # 1. Direct test case demo (matches the target API from the issue)
    run_direct_test_case_demo(client, space_id)

    # 2. Retrieval-only eval
    run_retrieval_only_eval(client, space_id)

    # 3. End-to-end RAG eval
    run_rag_eval(client, space_id)

    # 4. GoodMem vs baseline comparison
    run_comparison(client, space_id)

    print("\n" + "=" * 60)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
