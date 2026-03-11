# igoodmem_deepeval

An evaluation-focused integration between **GoodMem** and **DeepEval** that lets you benchmark GoodMem-powered retrieval and RAG pipelines using DeepEval's testing and metric framework.

This is *not* a DeepEval plugin or memory adapter. It's a thin Python package that:

- Uses **GoodMem** as the retriever/memory backend
- Converts GoodMem retrieval results into **DeepEval** `LLMTestCase` objects
- Supports **retrieval-only**, **end-to-end RAG**, and **component-level** evaluations
- Makes it easy to **benchmark GoodMem against other retrievers** using DeepEval's metrics
- Supports **reranking quality** and **metadata-filtered retrieval** evaluation

## Installation

```bash
pip install -e .
```

Requires Python 3.9+ and a running [GoodMem](https://docs.goodmem.ai) server.

## Quick Start

### 1. Set up environment variables

```bash
export GOODMEM_BASE_URL="http://localhost:8080"
export GOODMEM_API_KEY="gm_..."          # from `goodmem apikey list`
export OPENAI_API_KEY="sk-..."           # for DeepEval's LLM-as-judge metrics
```

### 2. Build a test case from GoodMem retrieval

```python
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase
from goodmem_deepeval import GoodMemEvalClient

client = GoodMemEvalClient(
    base_url="http://localhost:8080",
    api_key="gm_...",
)

# Create a space and store memories
space = client.create_space(
    space_name="rag-benchmark",
    embedder="bge-small-en",
    chunk_size=256,
    chunk_overlap=25,
)

client.create_memory(
    space=space.space_id,
    text_content="GoodMem supports semantic retrieval with optional reranking.",
    source="benchmark",
    author="eval-suite",
    tags="rag,retrieval",
)

# Retrieve and evaluate
retrieval = client.retrieve_memories(
    query="How does GoodMem improve retrieval quality?",
    spaces=[space.space_id],
    maximum_results=5,
)

test_case = LLMTestCase(
    input="How does GoodMem improve retrieval quality?",
    actual_output="GoodMem improves retrieval quality with semantic search and optional reranking.",
    expected_output="GoodMem uses semantic retrieval and can improve ranking quality through reranking.",
    retrieval_context=[item["content"] for item in retrieval["results"]],
)

evaluate(
    test_cases=[test_case],
    metrics=[
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
    ],
)
```

## Use Cases

### Evaluate a GoodMem-backed RAG pipeline

Build a full RAG pipeline with GoodMem retrieval and your own LLM generator, then evaluate it with DeepEval metrics.

```python
from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemRetriever,
    GoodMemRAGPipeline,
    evaluate_goodmem_rag,
)
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="gm_...")
retriever = GoodMemRetriever(client=client, spaces=["your-space-id"], maximum_results=5)

def my_generator(query: str, context: list[str]) -> str:
    # Call your LLM here (OpenAI, Anthropic, local model, etc.)
    return your_llm_call(query, context)

pipeline = GoodMemRAGPipeline(retriever=retriever, generator=my_generator)

result = evaluate_goodmem_rag(
    queries=["How does GoodMem work?", "What is semantic retrieval?"],
    pipeline=pipeline,
    metrics=[AnswerRelevancyMetric(threshold=0.7), FaithfulnessMetric(threshold=0.7)],
)
```

### Evaluate retrieval quality only (no LLM needed)

Test how well GoodMem retrieves relevant context without needing a generator LLM.

```python
from goodmem_deepeval import GoodMemEvalClient, evaluate_goodmem_retriever_only
from deepeval.metrics import ContextualRelevancyMetric, ContextualPrecisionMetric

client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="gm_...")

result = evaluate_goodmem_retriever_only(
    queries=["How does GoodMem work?", "What is reranking?"],
    client=client,
    spaces=["your-space-id"],
    metrics=[ContextualRelevancyMetric(threshold=0.5), ContextualPrecisionMetric(threshold=0.5)],
    maximum_results=5,
)
```

### Compare GoodMem against another retriever

Run the same queries through multiple RAG pipelines and compare scores side-by-side.

```python
from goodmem_deepeval import compare_pipelines
from deepeval.metrics import AnswerRelevancyMetric

results = compare_pipelines(
    queries=["How does GoodMem work?", "What is semantic search?"],
    pipelines={
        "goodmem": goodmem_pipeline,
        "baseline": your_other_pipeline,
    },
    metrics=[AnswerRelevancyMetric(threshold=0.7)],
)

for system_name, eval_result in results.items():
    print(f"{system_name}: {eval_result}")
```

### Evaluate reranking quality

Compare retrieval quality with and without a reranker to measure its impact.

```python
from goodmem_deepeval import GoodMemEvalClient, GoodMemRetriever, GoodMemRAGPipeline, compare_pipelines
from deepeval.metrics import ContextualRelevancyMetric

client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="gm_...")

retriever_no_rerank = GoodMemRetriever(client=client, spaces=["space-id"], maximum_results=5)
retriever_with_rerank = GoodMemRetriever(client=client, spaces=["space-id"], maximum_results=5, reranker="cross-encoder-reranker")

pipeline_no_rerank = GoodMemRAGPipeline(retriever=retriever_no_rerank, generator=my_generator)
pipeline_with_rerank = GoodMemRAGPipeline(retriever=retriever_with_rerank, generator=my_generator)

results = compare_pipelines(
    queries=["your queries here"],
    pipelines={"without_reranker": pipeline_no_rerank, "with_reranker": pipeline_with_rerank},
    metrics=[ContextualRelevancyMetric(threshold=0.5)],
)
```

### Evaluate metadata-filtered retrieval

Scope retrieval to specific metadata fields (source, author, tags, custom fields).

```python
from goodmem_deepeval import GoodMemEvalClient, GoodMemRetriever, evaluate_goodmem_retriever_only
from deepeval.metrics import ContextualRelevancyMetric

client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="gm_...")

# Only retrieve memories where source = "docs"
result = evaluate_goodmem_retriever_only(
    queries=["How does GoodMem handle search?"],
    client=client,
    spaces=["space-id"],
    metrics=[ContextualRelevancyMetric(threshold=0.5)],
    metadata_filter="CAST(val('$.source') AS TEXT) = 'docs'",
)
```

## GoodMem Client Operations

The `GoodMemEvalClient` wraps the GoodMem SDK with an evaluation-focused API:

```python
client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="gm_...")

# Spaces
space = client.create_space(space_name="my-space", embedder="bge-small-en", chunk_size=256, chunk_overlap=25)
spaces = client.list_spaces()

# Memories
memory = client.create_memory(space="space-id", text_content="...", source="docs", author="me", tags="a,b")
memories = client.list_memories(space_id="space-id")
memory = client.get_memory(memory_id="mem-id", include_content=True)
client.delete_memory(memory_id="mem-id")

# Retrieval with reranking and filtering
retrieval = client.retrieve_memories(
    query="search query",
    spaces=["space-id"],
    maximum_results=5,
    reranker="cross-encoder-reranker",       # optional reranker
    llm=None,                                 # optional LLM for post-processing
    relevance_threshold=0.5,                  # minimum relevance score
    llm_temperature=None,                     # LLM temperature
    chronological_resort=False,               # resort results by creation time
    metadata_filters={"filter": "CAST(val('$.source') AS TEXT) = 'docs'"},
)
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `GoodMemEvalClient` | Evaluation-focused wrapper over the GoodMem SDK |
| `GoodMemRetriever` | Retrieves chunks from GoodMem and returns structured `GoodMemChunk` objects |
| `GoodMemRAGPipeline` | Composes a retriever with a user-provided generator function |
| `GoodMemChunk` | Dataclass representing a retrieved chunk (content, score, metadata) |

### Evaluation Functions

| Function | Description |
|----------|-------------|
| `build_llm_test_case_from_goodmem()` | Build a DeepEval `LLMTestCase` from raw GoodMem retrieval results |
| `build_llm_test_case_from_pipeline()` | Run a pipeline and build a `LLMTestCase` from the output |
| `evaluate_goodmem_rag()` | Run end-to-end RAG evaluation over a list of queries |
| `evaluate_goodmem_retriever_only()` | Evaluate retrieval quality only (no generator needed) |
| `compare_pipelines()` | Compare multiple RAG pipelines on the same queries |

### Component-Level Tracing

| Function | Description |
|----------|-------------|
| `make_observed_retriever()` | Create a traced retriever for DeepEval observability |
| `make_observed_generator()` | Create a traced generator for DeepEval observability |
| `run_component_level_goodmem_rag_eval()` | Run a traced pipeline evaluation with DeepEval |

## Examples

| File | Description |
|------|-------------|
| `examples/smoke_goodmem_deepeval.py` | Minimal end-to-end smoke test |
| `examples/benchmark_goodmem_vs_baseline.py` | Compare GoodMem vs a baseline retriever |
| `examples/benchmark_reranking.py` | Compare retrieval with vs without reranking |
| `examples/metadata_filtered_retrieval.py` | Metadata-filtered retrieval evaluation |
| `examples/run_benchmark.py` | Comprehensive benchmark suite |

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 31 unit tests use mocks and do not require a running GoodMem server.

## Benchmark Results

Results from running `examples/run_benchmark.py` against a local GoodMem server with 5 ingested documents:

### End-to-End RAG (AnswerRelevancy + Faithfulness)

| Query | Answer Relevancy | Faithfulness |
|-------|-----------------|--------------|
| How does GoodMem improve retrieval quality? | 0.83 | 1.0 |
| What is RAG and how does it work? | 0.5 | 1.0 |
| How does metadata filtering work in GoodMem? | 0.5 | 1.0 |

### GoodMem vs Baseline (AnswerRelevancy)

| Query | GoodMem | Baseline |
|-------|---------|----------|
| How does GoodMem improve retrieval quality? | 1.0 | 0.0 |
| What is RAG and how does it work? | 1.0 | 0.0 |
| How does metadata filtering work in GoodMem? | 1.0 | 0.0 |

Faithfulness scores are perfect (1.0) across all queries, confirming that answers are fully grounded in retrieved context. GoodMem-backed pipelines significantly outperform the baseline on answer relevancy.

## Project Structure

```
src/goodmem_deepeval/
  __init__.py          # Public API exports
  client.py            # GoodMemEvalClient and config
  retriever.py         # GoodMemRetriever and GoodMemChunk
  pipeline.py          # GoodMemRAGPipeline
  deepeval_helpers.py  # DeepEval integration utilities
  comparison.py        # Multi-pipeline comparison
examples/              # Runnable example scripts
tests/                 # Unit tests (31 tests, all mocked)
```

## License

MIT
