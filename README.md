# igoodmem_deepeval

Evaluate and benchmark your GoodMem-powered RAG pipelines with [DeepEval](https://docs.confident-ai.com/).

If you're using GoodMem for retrieval in your AI application and want to know how well it's actually working — whether it's pulling the right documents, whether your LLM is hallucinating, or how GoodMem stacks up against other retrieval systems — this integration gives you the tools to measure all of that.

## What This Does

This package connects **GoodMem** (your retrieval/memory backend) to **DeepEval** (an LLM evaluation framework) so you can:

- **Score your retrieval quality** — Is GoodMem finding the right documents for a given question?
- **Score your full RAG pipeline** — When you combine GoodMem retrieval with an LLM, are the answers accurate and grounded?
- **Compare GoodMem against alternatives** — Run the same test queries through GoodMem and another retriever, get scores for both
- **Measure the impact of reranking** — Did enabling a reranker actually improve your results?
- **Test metadata-filtered retrieval** — When you filter by source, author, or tags, is the retrieval still good?

## When Would I Use This?

Here are some real scenarios where this integration helps:

**"I'm building a customer support chatbot and need to know if GoodMem is returning the right help articles."**
Load your support docs into GoodMem, write a list of common customer questions, and run `evaluate_goodmem_retriever_only()`. You'll get a contextual relevancy score for each query — if scores are low, you know to adjust your chunking, add a reranker, or clean up your data before going live.

**"My team is choosing between GoodMem and another vector database for our RAG app."**
Put the same documents in both systems, build a pipeline for each, and run `compare_pipelines()` with the same test queries. You get side-by-side scores so you can make a data-driven decision instead of guessing.

**"I turned on reranking and want to know if it's worth the extra latency."**
Create two retrievers — one with `reranker="cross-encoder-reranker"` and one without — and compare them. Hard numbers tell you if the quality improvement justifies the speed tradeoff.

**"I need to prove to stakeholders that our RAG system isn't hallucinating."**
Run `evaluate_goodmem_rag()` with `FaithfulnessMetric`. A score of 1.0 means the LLM's answers are fully grounded in retrieved context — zero hallucination. Run this regularly to catch regressions.

**"We have documents from different departments and I want to make sure filtered retrieval works well."**
Use `evaluate_goodmem_retriever_only()` with a metadata filter like `"CAST(val('$.source') AS TEXT) = 'engineering'"` to verify that scoped searches still return high-quality results.

**"I want automated quality checks in CI/CD — fail the build if RAG quality drops."**
Write a pytest test using `evaluate_goodmem_rag()` with threshold-based metrics (e.g., `AnswerRelevancyMetric(threshold=0.7)`). DeepEval fails the test if scores drop below your thresholds. Add it to your pipeline and quality regressions get caught automatically.

## Prerequisites

Before you start, you'll need:

1. **A running GoodMem server** — either self-hosted or cloud. See [GoodMem docs](https://docs.goodmem.ai) for setup.
2. **A GoodMem API key** — get one with `goodmem apikey list` or from your GoodMem dashboard.
3. **An OpenAI API key** — DeepEval uses GPT as an LLM judge to score your results. Get one at [platform.openai.com](https://platform.openai.com).
4. **Python 3.12+**

## Installation

**From GitHub:**
```bash
pip install git+https://github.com/PAIR-Systems-Inc/igoodmem_deepeval.git
```

**For local development:**
```bash
git clone https://github.com/PAIR-Systems-Inc/igoodmem_deepeval.git
cd igoodmem_deepeval
pip install -e .
```

**With benchmark dependencies (to run multi-provider comparisons):**
```bash
pip install -e ".[benchmark]"
```

## Step-by-Step Setup

### Step 1: Set your environment variables

```bash
export GOODMEM_BASE_URL="http://localhost:8080"    # your GoodMem server URL
export GOODMEM_API_KEY="gm_..."                    # your GoodMem API key
export OPENAI_API_KEY="sk-..."                     # your OpenAI API key (for DeepEval scoring)
```

### Step 2: Verify GoodMem is running

```bash
goodmem version           # verify CLI is installed
goodmem system health     # check server is running
goodmem apikey list       # verify your API key exists
```

### Step 3: Run the smoke test

```bash
python examples/smoke_goodmem_deepeval.py
```

This will:
- Connect to your GoodMem server
- Create a test space and add a sample document
- Retrieve context for a test query
- Run DeepEval metrics (Answer Relevancy + Contextual Relevancy)
- Print your scores

If you see scores printed, everything is working.

## Benchmark Results: GoodMem vs the Competition

We benchmarked GoodMem head-to-head against **Vectara**, **Weaviate**, and **Contextual AI** using a rigorous, standardized evaluation framework. The results speak for themselves.

### Test Setup

- **Dataset:** 200 Wikipedia articles from SQuAD 2.0 (~2,161 pages of content)
- **Queries:** 100 question-answer pairs with known ground truth
- **Generator:** GPT-4o-mini (identical across all providers)
- **Judge:** GPT-4.1 via DeepEval (identical scoring for all providers)
- **Metrics:** Answer Relevancy, Faithfulness (hallucination detection), Contextual Relevancy

### Results — Top-1 Retrieval

| Provider | Answer Relevancy | Faithfulness | Contextual Relevancy |
|----------|:---:|:---:|:---:|
| **GoodMem** | **0.954** | **0.855** | 0.453 |
| Vectara | 0.932 | 0.815 | **0.493** |
| Weaviate | 0.857 | 0.847 | 0.447 |
| Contextual AI | 0.912 | 0.735 | 0.262 |

### Results — Top-3 Retrieval

| Provider | Answer Relevancy | Faithfulness | Contextual Relevancy |
|----------|:---:|:---:|:---:|
| **GoodMem** | 0.889 | **0.762** | 0.283 |
| Vectara | **0.960** | 0.600 | 0.262 |
| Weaviate | 0.900 | 0.720 | **0.325** |
| Contextual AI | 0.873 | 0.500 | 0.148 |

### Key Takeaways

- **GoodMem delivers the highest Answer Relevancy at top-1 (0.954)** — it finds the most relevant content for precise, single-chunk retrieval.
- **GoodMem leads in Faithfulness across both top-1 and top-3** — answers grounded in GoodMem's retrieved context produce the fewest hallucinations. At top-3, GoodMem's Faithfulness (0.762) is **27% higher than Vectara** (0.600) and **52% higher than Contextual AI** (0.500).
- **Contextual AI struggles significantly at scale** — its Faithfulness drops to just 0.500 at top-3, meaning half the answers contain hallucinated content. Its Contextual Relevancy (0.148 at top-3) is the lowest by a wide margin.
- **GoodMem's intelligent chunking makes a difference** — while other providers use naive fixed-size character splitting, GoodMem uses a recursive hierarchical splitter that respects document structure (paragraph breaks, sentence boundaries). This produces cleaner, more semantically coherent chunks that lead to better downstream answers.

### How to Reproduce

```bash
pip install -e ".[benchmark]"

export GOODMEM_BASE_URL="http://localhost:8080"
export GOODMEM_API_KEY="gm_..."
export OPENAI_API_KEY="sk-..."
export VECTARA_API_KEY="..."
export WEAVIATE_URL="..."
export WEAVIATE_API_KEY="..."
export CONTEXTUAL_AI_API_KEY="..."

python -m examples.benchmark.run_benchmark \
  --num-articles 200 \
  --max-queries 100 \
  --top-k 1,3 \
  --providers goodmem,vectara,weaviate,contextual \
  --output results.json
```

---

## Usage Guide

### The basics: Connect, store, retrieve, evaluate

Every workflow follows the same pattern:

```python
from goodmem_deepeval import GoodMemEvalClient

# 1. Connect to GoodMem
client = GoodMemEvalClient(
    base_url="http://localhost:8080",
    api_key="gm_...",
)

# 2. Create a space (a collection for your documents)
space = client.create_space(
    space_name="my-knowledge-base",
    embedder="bge-small-en",       # the embedding model to use
    chunk_size=256,                 # how big each chunk should be
    chunk_overlap=25,               # overlap between chunks
)

# 3. Store your documents as memories
client.create_memory(
    space=space.space_id,
    text_content="Your document text goes here...",
    source="docs",                  # where this came from (optional)
    author="engineering",           # who wrote it (optional)
    tags="python,api",              # comma-separated tags (optional)
)

# 4. Retrieve context for a query
retrieval = client.retrieve_memories(
    query="How does the API work?",
    spaces=[space.space_id],
    maximum_results=5,
)

# 5. Build a DeepEval test case and evaluate
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="How does the API work?",
    actual_output="The API uses REST endpoints with JSON...",  # your LLM's answer
    retrieval_context=[item["content"] for item in retrieval["results"]],
)

evaluate(
    test_cases=[test_case],
    metrics=[
        AnswerRelevancyMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.7),
    ],
)
```

### Evaluate a full RAG pipeline

If you have a GoodMem retriever paired with an LLM generator, you can evaluate the entire pipeline at once:

```python
from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemRetriever,
    GoodMemRAGPipeline,
    evaluate_goodmem_rag,
)
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="gm_...")

# Set up the retriever
retriever = GoodMemRetriever(client=client, spaces=["your-space-id"], maximum_results=5)

# Provide your own LLM generator function
def my_generator(query: str, context: list[str]) -> str:
    # Call OpenAI, Anthropic, a local model, or whatever you use
    return your_llm_call(query, context)

# Combine retriever + generator into a pipeline
pipeline = GoodMemRAGPipeline(retriever=retriever, generator=my_generator)

# Evaluate with your test queries
result = evaluate_goodmem_rag(
    queries=[
        "How does the API work?",
        "What authentication methods are supported?",
        "How do I handle errors?",
    ],
    pipeline=pipeline,
    metrics=[
        AnswerRelevancyMetric(threshold=0.7),   # is the answer relevant to the question?
        FaithfulnessMetric(threshold=0.7),       # is the answer grounded in the context?
    ],
)
```

### Evaluate retrieval quality only (no LLM needed)

Sometimes you just want to know if GoodMem is pulling the right documents, without involving an LLM:

```python
from goodmem_deepeval import GoodMemEvalClient, evaluate_goodmem_retriever_only
from deepeval.metrics import ContextualRelevancyMetric, ContextualPrecisionMetric

client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="gm_...")

result = evaluate_goodmem_retriever_only(
    queries=["How does the API work?", "What is reranking?"],
    client=client,
    spaces=["your-space-id"],
    metrics=[
        ContextualRelevancyMetric(threshold=0.5),   # are the retrieved docs relevant?
        ContextualPrecisionMetric(threshold=0.5),    # are the most relevant docs ranked first?
    ],
    maximum_results=5,
)
```

### Compare GoodMem against another retriever

This is one of the most powerful features. Run the same queries through two different systems and compare:

```python
from goodmem_deepeval import compare_pipelines
from deepeval.metrics import AnswerRelevancyMetric

# goodmem_pipeline = your GoodMem RAG pipeline (see above)
# other_pipeline = your other RAG pipeline

results = compare_pipelines(
    queries=[
        "How does the API work?",
        "What authentication methods are supported?",
    ],
    pipelines={
        "goodmem": goodmem_pipeline,
        "other": other_pipeline,
    },
    metrics=[AnswerRelevancyMetric(threshold=0.7)],
)

# Print results per system
for system_name, eval_result in results.items():
    print(f"\n{system_name}:")
    print(eval_result)
```

Any pipeline that follows the signature `(query: str) -> (answer: str, context: list[str])` can be compared.

### Measure the impact of reranking

```python
from goodmem_deepeval import GoodMemEvalClient, GoodMemRetriever, GoodMemRAGPipeline, compare_pipelines
from deepeval.metrics import ContextualRelevancyMetric

client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="gm_...")

# Two retrievers: same config, but one has a reranker
retriever_basic = GoodMemRetriever(client=client, spaces=["space-id"], maximum_results=5)
retriever_reranked = GoodMemRetriever(client=client, spaces=["space-id"], maximum_results=5, reranker="cross-encoder-reranker")

pipeline_basic = GoodMemRAGPipeline(retriever=retriever_basic, generator=my_generator)
pipeline_reranked = GoodMemRAGPipeline(retriever=retriever_reranked, generator=my_generator)

results = compare_pipelines(
    queries=["your test queries"],
    pipelines={"without_reranker": pipeline_basic, "with_reranker": pipeline_reranked},
    metrics=[ContextualRelevancyMetric(threshold=0.5)],
)
```

### Evaluate metadata-filtered retrieval

If you tag your documents with metadata, you can evaluate filtered searches:

```python
from goodmem_deepeval import GoodMemEvalClient, evaluate_goodmem_retriever_only
from deepeval.metrics import ContextualRelevancyMetric

client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="gm_...")

# Only retrieve documents where source = "engineering-docs"
result = evaluate_goodmem_retriever_only(
    queries=["How does the API handle rate limiting?"],
    client=client,
    spaces=["space-id"],
    metrics=[ContextualRelevancyMetric(threshold=0.5)],
    metadata_filter="CAST(val('$.source') AS TEXT) = 'engineering-docs'",
)
```

## Understanding the Metrics

DeepEval provides several metrics. Here's what each one tells you:

| Metric | What it measures | When to use it |
|--------|-----------------|----------------|
| **AnswerRelevancyMetric** | Is the LLM's answer relevant to the question? | End-to-end RAG evaluation |
| **FaithfulnessMetric** | Is the answer grounded in the retrieved context (no hallucination)? | Checking for hallucination |
| **ContextualRelevancyMetric** | Are the retrieved documents relevant to the query? | Retrieval quality testing |
| **ContextualPrecisionMetric** | Are the most relevant documents ranked highest? | Ranking quality testing |

Each metric takes a `threshold` (0.0 to 1.0). A score above the threshold passes; below it fails.

## GoodMem Client Operations

The `GoodMemEvalClient` provides all the GoodMem operations you need for evaluation:

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

# Retrieval (with optional reranking and filtering)
retrieval = client.retrieve_memories(
    query="search query",
    spaces=["space-id"],
    maximum_results=5,
    reranker="cross-encoder-reranker",       # optional: reranker to use
    relevance_threshold=0.5,                  # optional: minimum relevance score
    metadata_filters={"filter": "CAST(val('$.source') AS TEXT) = 'docs'"},  # optional: filter expression
)
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `GoodMemEvalClient` | Connects to GoodMem and provides space/memory/retrieval operations |
| `GoodMemRetriever` | Wraps the client for structured retrieval with `GoodMemChunk` objects |
| `GoodMemRAGPipeline` | Combines a retriever with your LLM generator into an evaluatable pipeline |
| `GoodMemChunk` | A retrieved chunk with `content`, `score`, `memory_id`, `metadata` fields |

### Evaluation Functions

| Function | Description |
|----------|-------------|
| `evaluate_goodmem_rag()` | Evaluate a full RAG pipeline (retrieval + generation) |
| `evaluate_goodmem_retriever_only()` | Evaluate just the retrieval quality (no LLM needed) |
| `compare_pipelines()` | Compare multiple pipelines (GoodMem vs others) on the same queries |
| `build_llm_test_case_from_goodmem()` | Build a DeepEval `LLMTestCase` from raw retrieval results |
| `build_llm_test_case_from_pipeline()` | Run a pipeline and build a `LLMTestCase` from the output |

### Component-Level Tracing

For advanced users who want DeepEval observability per-component:

| Function | Description |
|----------|-------------|
| `make_observed_retriever()` | Create a traced retriever for DeepEval observability |
| `make_observed_generator()` | Create a traced generator for DeepEval observability |
| `run_component_level_goodmem_rag_eval()` | Run a traced pipeline evaluation |

## Examples

| File | What it does |
|------|-------------|
| `examples/smoke_goodmem_deepeval.py` | Minimal end-to-end test — good place to start |
| `examples/benchmark_goodmem_vs_baseline.py` | Compare GoodMem against a dummy baseline |
| `examples/benchmark_reranking.py` | Measure impact of reranking on retrieval quality |
| `examples/metadata_filtered_retrieval.py` | Evaluate metadata-filtered retrieval |
| `examples/hallucination_detection.py` | Detect hallucination — compares faithful vs hallucinating generators |
| `examples/openai_rag_eval.py` | Real LLM end-to-end RAG eval using OpenAI GPT-4o-mini |
| `examples/retrieval_tuning_eval.py` | Retrieval tuning — compare broad vs precise retrieval settings |
| `examples/benchmark/run_benchmark.py` | **Full multi-provider benchmark** — GoodMem vs Vectara vs Weaviate vs Contextual AI |

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 31 unit tests use mocks and do not require a running GoodMem server or any API keys.

## Project Structure

```
src/goodmem_deepeval/
  __init__.py          # Public API exports
  client.py            # GoodMemEvalClient — connects to GoodMem
  retriever.py         # GoodMemRetriever — structured retrieval
  pipeline.py          # GoodMemRAGPipeline — retriever + generator
  deepeval_helpers.py  # DeepEval integration — test case builders and eval functions
  comparison.py        # compare_pipelines() — multi-system benchmarking
examples/
  benchmark/           # Multi-provider benchmark framework
    providers/         # GoodMem, Vectara, Weaviate, Contextual AI provider plugins
    run_benchmark.py   # CLI entry point for running benchmarks
  ...                  # Additional example scripts (smoke tests, reranking, etc.)
tests/                 # Unit tests (31 tests, all mocked)
```

## License

MIT
