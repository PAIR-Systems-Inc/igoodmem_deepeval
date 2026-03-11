from __future__ import annotations

"""
Benchmark retrieval quality with and without a reranker.

Compares two GoodMemRetriever configurations — one using a reranker and one
without — using DeepEval's ContextualRelevancyMetric.
"""

import os

from deepeval.metrics import ContextualRelevancyMetric

from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemRetriever,
    GoodMemRAGPipeline,
    compare_pipelines,
)


def main() -> None:
    base_url = os.environ.get("GOODMEM_BASE_URL", "http://localhost:8080")
    api_key = os.environ.get("GOODMEM_API_KEY")
    reranker_id = os.environ.get("GOODMEM_RERANKER_ID", "cross-encoder-reranker")
    if not api_key:
        raise RuntimeError("GOODMEM_API_KEY is not set in the environment.")

    client = GoodMemEvalClient(base_url=base_url, api_key=api_key)

    spaces = client.list_spaces()
    space = next((s for s in spaces if s.name == "rag-benchmark"), None)
    if not space:
        raise RuntimeError("Expected a space named 'rag-benchmark'; run the smoke example first.")

    # Retriever WITHOUT reranker
    retriever_no_rerank = GoodMemRetriever(
        client=client,
        spaces=[space.space_id],
        maximum_results=5,
    )

    # Retriever WITH reranker
    retriever_with_rerank = GoodMemRetriever(
        client=client,
        spaces=[space.space_id],
        maximum_results=5,
        reranker=reranker_id,
    )

    def generator(query: str, context: list[str]) -> str:
        if context:
            return f"Answer based on: {context[0]}"
        return "No context retrieved."

    pipeline_no_rerank = GoodMemRAGPipeline(retriever=retriever_no_rerank, generator=generator)
    pipeline_with_rerank = GoodMemRAGPipeline(retriever=retriever_with_rerank, generator=generator)

    queries = [
        "How does GoodMem improve retrieval quality?",
        "What is semantic search?",
        "How does reranking work?",
    ]

    metrics = [ContextualRelevancyMetric(threshold=0.5)]

    results = compare_pipelines(
        queries=queries,
        pipelines={
            "without_reranker": pipeline_no_rerank,
            "with_reranker": pipeline_with_rerank,
        },
        metrics=metrics,
    )

    for name, result in results.items():
        print(f"\n=== {name} ===")
        print(result)


if __name__ == "__main__":
    main()
