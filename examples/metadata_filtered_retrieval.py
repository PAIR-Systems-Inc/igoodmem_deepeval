from __future__ import annotations

"""
Demonstrate metadata-filtered retrieval and evaluation.

Creates memories with different metadata tags, then retrieves using a filter
expression and evaluates with DeepEval.
"""

import os

from deepeval.metrics import ContextualRelevancyMetric

from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemRetriever,
    GoodMemRAGPipeline,
    evaluate_goodmem_rag,
)


def main() -> None:
    base_url = os.environ.get("GOODMEM_BASE_URL", "http://localhost:8080")
    api_key = os.environ.get("GOODMEM_API_KEY")
    if not api_key:
        raise RuntimeError("GOODMEM_API_KEY is not set in the environment.")

    client = GoodMemEvalClient(base_url=base_url, api_key=api_key)

    # Reuse 'rag-benchmark' space
    spaces = client.list_spaces()
    space = next((s for s in spaces if s.name == "rag-benchmark"), None)
    if not space:
        raise RuntimeError("Expected a space named 'rag-benchmark'; run the smoke example first.")
    space_id = space.space_id

    # Add memories with different sources
    client.create_memory(
        space=space_id,
        text_content="GoodMem uses vector embeddings for fast semantic search.",
        source="docs",
        author="engineering",
        tags="search,embeddings",
    )
    client.create_memory(
        space=space_id,
        text_content="The marketing team uses GoodMem for customer support retrieval.",
        source="marketing",
        author="marketing-team",
        tags="support,customers",
    )

    # Retrieve only docs-sourced memories using a metadata filter
    retriever = GoodMemRetriever(
        client=client,
        spaces=[space_id],
        maximum_results=5,
        metadata_filter="CAST(val('$.source') AS TEXT) = 'docs'",
    )

    def generator(query: str, context: list[str]) -> str:
        if context:
            return f"Based on docs: {context[0]}"
        return "No docs-sourced context found."

    pipeline = GoodMemRAGPipeline(retriever=retriever, generator=generator)

    queries = ["How does GoodMem handle search?"]
    metrics = [ContextualRelevancyMetric(threshold=0.3)]

    result = evaluate_goodmem_rag(queries=queries, pipeline=pipeline, metrics=metrics)
    print(result)


if __name__ == "__main__":
    main()
