# GoodMem Integration Comparison Report

## Overview

| Dimension | `igoodmem_deepeval` (A) | `deepeval/integrations/goodmem` (B) |
|---|---|---|
| Location | Standalone package | Inside deepeval repo |
| Architecture | 5-layer (SDK → Client → Retriever → Pipeline → Helpers) | 2-layer (Client → Types) |
| External deps | `goodmem-client>=1.5.12` + `typing-extensions` | `requests` only |
| Lines of code | ~600 Python | ~700 Python |
| Tests | 31 unit tests (all mocked) | 10 integration tests (all live) |
| Examples | 9 runnable scripts | 0 |
| Documentation | 500-line README | None |

---

## Integration A — `igoodmem_deepeval`

### Pros

**1. Richest RAG evaluation surface in either integration.**
Provides `evaluate_goodmem_rag()`, `evaluate_goodmem_retriever_only()`, `run_component_level_goodmem_rag_eval()` — users can evaluate end-to-end RAG, retrieval-only, or per-component with a single call. B has none of this.

**2. `compare_pipelines()` is genuinely powerful.**
Side-by-side scoring of GoodMem vs any other retriever (Vectara, dummy baseline, different configs). This is a killer feature for adoption — users want to prove GoodMem is better than what they have.

**3. `@observe` component tracing.**
Wraps retriever and generator with DeepEval's tracing decorators so you get per-span metrics in the DeepEval dashboard. B doesn't integrate with DeepEval observability at all.

**4. Exemplary documentation and examples.**
500-line README with a metrics guide, API reference, benchmark result tables, and 9 runnable scripts covering hallucination detection, reranking impact, metadata filtering, cross-system comparison, and OpenAI end-to-end. This is production-quality DX.

**5. Unit test coverage.**
31 tests covering all code paths without needing a live server. CI pipelines can run these on every PR.

**6. `GoodMemChunk` type is ergonomic.**
Exposes `.source`, `.author`, `.tags` as properties parsed from metadata. Much friendlier than raw dicts.

**7. Retrieval-only eval.**
`evaluate_goodmem_retriever_only()` scores `ContextualRelevancyMetric` and `ContextualPrecisionMetric` without requiring an LLM generator at all — critical for teams still building their RAG pipeline.

### Cons

**1. Fatal deployment blocker: depends on `goodmem-client` SDK.**
This is an external package not already in deepeval's dependency tree. Shipping this as a deepeval integration forces all deepeval users to install `goodmem-client` or adds it as an optional dep with fragile version pinning. If the SDK releases a breaking change, the integration silently breaks. B avoids this entirely with direct HTTP.

**2. Wrong venue — it's a standalone package, not a deepeval integration.**
Cannot be imported as `from deepeval.integrations.goodmem import ...`. To ship this in deepeval, every file would need to be restructured, the SDK dependency removed, and all the SDK abstractions rewritten. That's effectively a rewrite.

**3. SDK double-wrapping adds fragility.**
`GoodMemRetriever` → `GoodMemEvalClient` → `goodmem_client` SDK. Three layers, two of which are outside this codebase. When the SDK changes its response shapes or event stream format, A breaks and the fix requires both a PR here and waiting for a goodmem-client release.

**4. No real-server tests.**
All 31 unit tests use `MagicMock`. They test Python wiring, not actual API behavior. The smoke test examples work (as shown in the results), but there's no pytest test suite that runs against a real server. B has 10 passing live tests with documented results.

**5. No file upload support.**
`create_memory()` only accepts `text_content: str`. B supports PDF, images, Word docs, spreadsheets, and 17 file extensions. For a production knowledge base, file upload is essential.

**6. Metadata API is prescriptive.**
Enforces `source`, `author`, `tags` top-level params. The real GoodMem API has a generic metadata dict. This opinionated schema could conflict with how the actual server stores metadata.

**7. `evaluate_goodmem_retriever_only()` leaks abstraction.**
Internally builds a fake pipeline with a dummy generator just to drive the evaluation loop. Users shouldn't need a generator for retrieval-only evaluation — the abstraction is wrong.

**8. `compare_pipelines()` wraps DeepEval's `evaluate()` in a thin dict.**
Returns `dict[name → EvaluationResult]` but doesn't provide any aggregated scoring, delta visualization, or ranking. Users still have to manually parse results to compare.

---

## Integration B — `deepeval/integrations/goodmem`

### Pros

**1. Zero external dependencies.**
Uses only `requests`, `json`, `base64`, `os`, `time`, `typing` — all already available in a deepeval install. No new transitive deps, no version conflicts, ships immediately.

**2. Lives in the right place.**
`deepeval/integrations/goodmem/` follows the exact same pattern as `deepeval/integrations/langchain/`, `llama_index/`, `crewai/`. Zero restructuring needed to merge into main.

**3. Direct HTTP client = full API control.**
No SDK intermediary. When the GoodMem API adds a parameter or changes a response field, the fix is one line in `client.py`. No waiting for SDK releases.

**4. Real-server tested.**
All 10 tests passed against a live GoodMem server with documented results. We know exactly what breaks (OpenAI Ada embedder fails, text memory content_error is expected behavior) and what works.

**5. File upload support.**
Handles 17 MIME types — PDF, PNG, JPEG, GIF, WEBP, TXT, HTML, Markdown, CSV, JSON, XML, DOC/DOCX, XLS/XLSX, PPT/PPTX. B64-encodes files and sends correct MIME type. A has none of this.

**6. Streaming/NDJSON response parsing.**
`_parse_retrieval_response()` correctly handles the server's NDJSON/SSE stream. This is non-trivial to get right and it works.

**7. Model discovery.**
`list_embedders()`, `list_rerankers()`, `list_llms()` expose what's available on the server. Users don't need to know IDs in advance. A's client doesn't expose this.

**8. `retrieve_as_context()`.**
One-liner that returns `List[str]` ready for `LLMTestCase(retrieval_context=...)`. Clean, explicit integration point.

**9. Space reuse by name.**
`create_space()` silently reuses an existing space with the same name. Idempotent setup — safe to run in tests repeatedly.

**10. `wait_for_indexing` with polling.**
60-second timeout with 5-second intervals. Handles the async nature of memory processing correctly. A delegates this to the SDK.

### Cons

**1. No evaluation layer at all.**
The biggest gap. Users must manually call `retrieve_as_context()`, build `LLMTestCase`, import metrics, call `evaluate()`. Every user writes the same boilerplate. A provides 5 high-level functions that eliminate this.

**2. No pipeline abstraction.**
No `GoodMemRAGPipeline`, no way to define "retriever + generator" as a reusable unit. Users can't call `compare_pipelines()` because it doesn't exist.

**3. No examples.**
Zero runnable scripts. Users have no starting point. The only reference is the test file, which isn't written for learning.

**4. No documentation.**
No README, no docstrings explaining parameters, no metrics guide, no use case walkthrough. Unusable without reading the source.

**5. No unit tests.**
Every test requires a live server. CI pipelines can't run this on PRs without test infrastructure. A's mocked tests are better for pure regression testing.

**6. `list_spaces()` returns raw `List[Dict]`.**
Inconsistent with the typed approach everywhere else. Should return `List[GoodMemSpace]`.

**7. No `@observe` tracing.**
No integration with DeepEval's observability layer. A supports per-component metrics in the dashboard.

**8. No comparison utilities.**
No built-in way to benchmark GoodMem vs a baseline. A's `compare_pipelines()` is a strong differentiator.

**9. Metadata is unstructured.**
`create_memory(metadata={"key": "val"})` passes arbitrary dicts. A's enforced `source`/`author`/`tags` schema makes chunks more queryable via metadata filters (even if it's opinionated).

**10. No retrieval-only evaluation helpers.**
Users who want to score retrieval quality alone must manually construct `LLMTestCase` with only `retrieval_context` and no `actual_output`.

---

## Scoring (out of 100)

Criteria and weights for production readiness on deepeval:

| Criterion | Weight | A Score | B Score |
|---|---|---|---|
| **Deployment fit** — zero deps, lives in deepeval repo, ships without restructuring | 25% | 15/25 | 24/25 |
| **Functionality** — completeness of features users need | 22% | 20/22 | 12/22 |
| **Code quality** — type safety, error handling, clean abstractions | 18% | 13/18 | 16/18 |
| **Testing** — real coverage, CI-friendly | 15% | 8/15 | 11/15 |
| **Documentation & DX** — README, examples, ease of use | 12% | 11/12 | 2/12 |
| **DeepEval ecosystem depth** — metrics, tracing, test cases | 8% | 7/8 | 4/8 |

| Integration | Total |
|---|---|
| **A — `igoodmem_deepeval`** | **74 / 100** |
| **B — `deepeval/integrations/goodmem`** | **69 / 100** |

---

## Verdict

Neither integration is ready to ship alone. They solve opposite halves of the problem:

- **A** has the right *what* (evaluation abstractions, pipeline, comparison, docs) but the wrong *where* (standalone package with SDK dependency)
- **B** has the right *where* (in deepeval, zero deps, direct HTTP, real tests) but is missing the evaluation layer

**The winning integration = B's architecture + A's evaluation helpers.**

Concretely: take B as the base (it stays in deepeval, uses requests, is already tested), and port A's `retriever.py`, `pipeline.py`, `deepeval_helpers.py`, and `comparison.py` into it — rewriting them to use `GoodMemClient` directly instead of through the SDK. Add A's examples and README. That produces a 90+ integration.
