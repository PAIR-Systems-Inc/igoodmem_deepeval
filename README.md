## igoodmem_deepeval

`igoodmem_deepeval` is a small, evaluation‑focused integration between **GoodMem** and **DeepEval**.

It is *not* a DeepEval plugin or a memory adapter inside DeepEval itself. Instead, it provides a thin Python package that:

- Uses **GoodMem** as the retriever / memory backend.
- Converts GoodMem retrieval results into **DeepEval** `LLMTestCase`s.
- Supports both **retrieval‑only** and **end‑to‑end RAG** evaluations.
- Makes it easy to **benchmark GoodMem‑backed pipelines against other retrievers** using DeepEval’s existing metrics.

### Key pieces

All core code lives under `src/goodmem_deepeval/`:

- **`client.py`**
  - `GoodMemEvalConfig`: holds `base_url`, `api_key`, and optional timeout.
  - `GoodMemEvalClient`: eval‑oriented wrapper over `goodmem-client`:
    - Spaces: `create_space`, `list_spaces`.
    - Memories: `create_memory`, `list_memories`, `get_memory`, `delete_memory`.
    - Retrieval: `retrieve_memories(...)` using `MemoryStreamClient.retrieve_memory_stream`, returning a simple, DeepEval‑friendly dict.

- **`retriever.py`**
  - `GoodMemChunk`: lightweight representation of a retrieved chunk (content, score, ids, metadata).
  - `GoodMemRetriever`: wraps `GoodMemEvalClient.retrieve_memories(...)` and exposes:
    - `retrieve(query: str) -> list[GoodMemChunk]`
    - `to_text_list(chunks) -> list[str]` for DeepEval `retrieval_context`.

- **`pipeline.py`**
  - `GoodMemRAGPipeline`: composes a `GoodMemRetriever` with a user‑provided generator:
    - `__call__(query) -> (answer, retrieval_context)`
    - Returns exactly what DeepEval’s RAG quickstart expects.

- **`deepeval_helpers.py`**
  - End‑to‑end:
    - `build_llm_test_case_from_goodmem(...)`
    - `build_llm_test_case_from_pipeline(...)`
    - `evaluate_goodmem_rag(...)` – run full RAG evals via `deepeval.evaluate`.
  - Retrieval‑only:
    - `evaluate_goodmem_retriever_only(...)` – for metrics like `ContextualRelevancyMetric`, `ContextualPrecisionMetric`, etc.
  - Component‑level / tracing:
    - `make_observed_retriever(...)`, `make_observed_generator(...)`
    - `run_component_level_goodmem_rag_eval(...)` – wire GoodMem into DeepEval’s tracing model.

- **`comparison.py`**
  - `compare_pipelines(queries, pipelines, metrics)`:
    - Runs multiple RAG pipelines (e.g. GoodMem vs baseline) on the same queries.
    - Builds `LLMTestCase`s and calls `deepeval.evaluate(...)` per system.
    - Returns a dict of `EvaluationResult` per system name.

### Examples

All examples live in `examples/` and are meant to be minimal but executable.

#### Smoke test: GoodMem‑backed RAG pipeline

File: `examples/smoke_goodmem_deepeval.py`

- Connects to a running GoodMem server.
- Finds an existing embedder and creates/reuses a `rag-benchmark` space.
- Inserts a small test memory:
  > “GoodMem supports semantic retrieval with optional reranking.”
- Builds:
  - `GoodMemRetriever` over that space.
  - `GoodMemRAGPipeline` with a simple generator.
- Runs DeepEval with:
  - `AnswerRelevancyMetric`
  - `ContextualRelevancyMetric`
- For the query:
  > “How does GoodMem improve retrieval quality?”

With a valid GoodMem API key and OpenAI key configured, this script runs end‑to‑end and produces DeepEval scores (in our tests, contextual relevancy is 1.0 and answer relevancy behaves as expected based on the generator).

#### Benchmark: GoodMem vs baseline retriever

File: `examples/benchmark_goodmem_vs_baseline.py`

- Reuses the `rag-benchmark` space and GoodMem retriever.
- Defines:
  - A **GoodMem‑backed** RAG pipeline (GoodMem + simple generator).
  - A **baseline** pipeline that returns `"Baseline answer"` and `["Baseline retrieved context"]`.
- Runs both on a small query set:
  - “How does GoodMem improve retrieval quality?”
  - “What is GoodMem used for?”
- Evaluates each system with:
  - `AnswerRelevancyMetric`
  - `ContextualRelevancyMetric`
- Uses `compare_pipelines(...)` to return per‑system `EvaluationResult`s and prints them.

This demonstrates how to benchmark GoodMem‑backed RAG pipelines against another retriever using DeepEval’s existing test‑case and metrics APIs.

### Running locally

1. **Install in editable mode**

   From the `igoodmem_deepeval` project root:
```bash
   cd igoodmem_deepeval
   pip install -e .
```

2. **Ensure GoodMem is running and you have an API key**

   Use the GoodMem CLI to verify the server and list your keys:
```bash
   goodmem system info
   goodmem apikey list
```
   Pick an active key (prefix `gm_...`) from the list.

3. **Export environment variables for GoodMem + DeepEval**
```bash
   export GOODMEM_BASE_URL="http://localhost:8080"   # or your GoodMem REST URL
   export GOODMEM_API_KEY="gm_..."                   # from `goodmem apikey list`
   export OPENAI_API_KEY="sk-..."                    # for DeepEval's LLM-as-judge metrics
```

4. **Run the smoke example**
```bash
   python examples/smoke_goodmem_deepeval.py
```
   This will:
   - Discover an embedder via the GoodMem API
   - Create/reuse a `rag-benchmark` space
   - Insert a test memory
   - Run a GoodMem-backed RAG pipeline through DeepEval's RAG metrics

5. **Run the benchmark example** (optional)
```bash
   python examples/benchmark_goodmem_vs_baseline.py
```
   This compares a GoodMem-backed pipeline against a simple baseline retriever using the same DeepEval metrics.

From here you can iterate on:
- The generator function (swap in your real LLM)
- The metrics (add/remove DeepEval metrics)
- The queries/datasets you want to use for GoodMem vs other retriever benchmarks
