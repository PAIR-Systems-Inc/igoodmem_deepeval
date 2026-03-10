import os
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric
from goodmem_client.api.embedders_api import EmbeddersApi
from goodmem_client.api_client import ApiClient
from goodmem_client.configuration import Configuration
from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemEvalConfig,
    GoodMemRetriever,
    GoodMemRAGPipeline,
    evaluate_goodmem_rag,
)

BASE_URL = os.getenv("GOODMEM_BASE_URL", "http://localhost:8080")  # GoodMem REST endpoint
API_KEY = os.getenv("GOODMEM_API_KEY")             # paste your API key here
if not API_KEY:
    raise ValueError("GOODMEM_API_KEY is not set in the environment.")

def main():
    # 1) Wire client
    config = GoodMemEvalConfig(base_url=BASE_URL, api_key=API_KEY)
    client = GoodMemEvalClient(config)

    # 1b) Pick an existing embedder (required by the server)
    cfg = Configuration()
    cfg.host = BASE_URL
    cfg.api_key = {"ApiKeyAuth": API_KEY}
    embedder_client = ApiClient(configuration=cfg)
    embedders_api = EmbeddersApi(embedder_client)
    embedders = embedders_api.list_embedders().embedders
    if not embedders:
        raise RuntimeError("No embedders found in GoodMem; create one with `goodmem embedder` first.")
    embedder_id = embedders[0].embedder_id

    # 2) Get or create a space for the benchmark, attached to the chosen embedder
    spaces = client.list_spaces()
    existing = next((s for s in spaces if s.name == "rag-benchmark"), None)
    if existing:
        space = existing
    else:
        space = client.create_space(space_name="rag-benchmark", embedder_id=embedder_id)
    space_id = space.space_id

    # 3) Add one memory into that space
    client.create_memory(
        space_id=space_id,
        text_content="GoodMem supports semantic retrieval with optional reranking.",
        metadata={"source": "benchmark", "author": "eval-suite"},
    )

    # 4) Build retriever and simple generator
    retriever = GoodMemRetriever(
        client=client,
        spaces=[space_id],
        maximum_results=5,
    )

    def simple_generator(query: str, context: list[str]) -> str:
        # In a real app, call an LLM here; for now, echo some context.
        if context:
            return f"GoodMem-based answer: {context[0]}"
        return "No context retrieved."

    pipeline = GoodMemRAGPipeline(retriever=retriever, generator=simple_generator)

    # 5) Run a tiny DeepEval run
    metrics = [
        AnswerRelevancyMetric(threshold=0.1),
        ContextualRelevancyMetric(threshold=0.1),
    ]

    queries = ["How does GoodMem improve retrieval quality?"]

    result = evaluate_goodmem_rag(queries=queries, pipeline=pipeline, metrics=metrics)
    print(result)

if __name__ == "__main__":
    main()