from __future__ import annotations

"""
Example script showing how to benchmark a GoodMem-backed RAG pipeline
against a baseline pipeline using DeepEval metrics.

This is intentionally minimal and uses a dummy baseline; in real usage
you would replace `baseline_pipeline` with your existing RAG system.
"""

from typing import List, Tuple
import os

from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric

from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemEvalConfig,
    GoodMemRetriever,
    GoodMemRAGPipeline,
    compare_pipelines,
)


def main() -> None:
    base_url = os.environ.get("GOODMEM_BASE_URL", "http://localhost:8080")
    api_key = os.environ.get("GOODMEM_API_KEY")
    if not api_key:
        raise RuntimeError("GOODMEM_API_KEY is not set in the environment.")

    config = GoodMemEvalConfig(
        base_url=base_url,
        api_key=api_key,
    )
    client = GoodMemEvalClient(config)

    # Assume you have already created a space and ingested memories.
    # Here we reuse the 'rag-benchmark' space created by the smoke example.
    spaces = client.list_spaces()
    space = next((s for s in spaces if s.name == "rag-benchmark"), None)
    if not space:
        raise RuntimeError("Expected a space named 'rag-benchmark'; run the smoke example first.")

    retriever = GoodMemRetriever(
        client=client,
        spaces=[space.space_id],
        maximum_results=5,
    )

    def goodmem_generator(query: str, context: List[str]) -> str:
        # Replace with a real LLM call that uses GoodMem context.
        return "GoodMem-backed answer"

    goodmem_pipeline = GoodMemRAGPipeline(
        retriever=retriever,
        generator=goodmem_generator,
    )

    # Dummy baseline pipeline; replace with your own implementation.
    def baseline_pipeline(query: str) -> Tuple[str, List[str]]:
        return "Baseline answer", ["Baseline retrieved context"]

    queries = [
        "How does GoodMem improve retrieval quality?",
        "What is GoodMem used for?",
    ]

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.7),
    ]

    results = compare_pipelines(
        queries=queries,
        pipelines={
            "goodmem": goodmem_pipeline,
            "baseline": baseline_pipeline,
        },
        metrics=metrics,
    )

    print(results)


if __name__ == "__main__":
    main()

