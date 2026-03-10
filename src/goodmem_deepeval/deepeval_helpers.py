from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span

from .client import GoodMemEvalClient
from .pipeline import GoodMemRAGPipeline, RAGPipeline
from .retriever import GoodMemRetriever


# ---------------------------------------------------------------------------
# End-to-end helpers
# ---------------------------------------------------------------------------


def build_llm_test_case_from_goodmem(
    query: str,
    actual_output: Optional[str],
    retrieval: Dict[str, Any],
    expected_output: Optional[str] = None,
) -> LLMTestCase:
    """
    Build a DeepEval LLMTestCase directly from a GoodMemEvalClient.retrieve_memories
    result dictionary plus an optional model output.
    """
    results: Sequence[Dict[str, Any]] = retrieval.get("results", []) or []
    retrieval_context: List[str] = [item.get("content", "") for item in results]

    return LLMTestCase(
        input=query,
        actual_output=actual_output or "",
        retrieval_context=retrieval_context,
        expected_output=expected_output,
    )


def build_llm_test_case_from_pipeline(
    query: str,
    pipeline: GoodMemRAGPipeline,
    expected_output: Optional[str] = None,
) -> LLMTestCase:
    """
    Call a GoodMemRAGPipeline and convert its output into an LLMTestCase.
    """
    answer, retrieval_context = pipeline(query)
    return LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=retrieval_context,
        expected_output=expected_output,
    )


def evaluate_goodmem_rag(
    queries: Sequence[str],
    pipeline: GoodMemRAGPipeline,
    metrics: Sequence[Any],
    expected_outputs: Optional[Mapping[str, str]] = None,
) -> Any:
    """
    Convenience wrapper to run end-to-end RAG evaluation for a GoodMem-backed
    pipeline over a list of queries.
    """
    test_cases: List[LLMTestCase] = []
    expected_outputs = expected_outputs or {}
    for q in queries:
        exp = expected_outputs.get(q)
        test_cases.append(build_llm_test_case_from_pipeline(q, pipeline, exp))
    return evaluate(test_cases=test_cases, metrics=list(metrics))


def evaluate_goodmem_retriever_only(
    queries: Sequence[str],
    client: GoodMemEvalClient,
    spaces: Iterable[str],
    metrics: Sequence[Any],
    maximum_results: int = 5,
    relevance_threshold: Optional[float] = None,
    reranker_id: Optional[str] = None,
    metadata_filter: Optional[str] = None,
) -> Any:
    """
    Evaluate only the retrieval component using DeepEval's retrieval-oriented
    metrics such as ContextualRelevancyMetric, ContextualPrecisionMetric, etc.
    """
    retriever = GoodMemRetriever(
        client=client,
        spaces=spaces,
        maximum_results=maximum_results,
        relevance_threshold=relevance_threshold,
        reranker_id=reranker_id,
        metadata_filter=metadata_filter,
    )

    from .retriever import GoodMemChunk  # for type hints in comprehension

    test_cases: List[LLMTestCase] = []
    for q in queries:
        chunks: Sequence[GoodMemChunk] = retriever.retrieve(q)
        retrieval_context = retriever.to_text_list(chunks)
        test_cases.append(
            LLMTestCase(
                input=q,
                actual_output="",
                retrieval_context=retrieval_context,
            )
        )
    return evaluate(test_cases=test_cases, metrics=list(metrics))


# ---------------------------------------------------------------------------
# Component-level tracing helpers
# ---------------------------------------------------------------------------


@dataclass
class ObservedRetrieverConfig:
    retriever: GoodMemRetriever


def make_observed_retriever(metrics: Sequence[Any]) -> Callable[[ObservedRetrieverConfig, str], List[str]]:
    """
    Factory for an @observe-wrapped retriever function that can be used
    inside traced RAG pipelines.
    """

    @observe(metrics=list(metrics))
    def _observed(config: ObservedRetrieverConfig, query: str) -> List[str]:
        chunks = config.retriever.retrieve(query)
        retrieval_context = config.retriever.to_text_list(chunks)
        test_case = LLMTestCase(
            input=query,
            retrieval_context=retrieval_context,
            actual_output="",
        )
        update_current_span(test_case=test_case)
        return retrieval_context

    return _observed


def make_observed_generator(metrics: Sequence[Any]) -> Callable[[Callable[[str, Sequence[str]], str], str, List[str]], str]:
    """
    Factory for an @observe-wrapped generator function that can be used
    inside traced RAG pipelines.
    """

    @observe(metrics=list(metrics))
    def _observed(
        generator_fn: Callable[[str, Sequence[str]], str],
        query: str,
        retrieval_context: List[str],
    ) -> str:
        answer = generator_fn(query, retrieval_context)
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=retrieval_context,
        )
        update_current_span(test_case=test_case)
        return answer

    return _observed


def run_component_level_goodmem_rag_eval(
    dataset: EvaluationDataset,
    retriever: GoodMemRetriever,
    generator_fn: Callable[[str, Sequence[str]], str],
    retriever_metrics: Sequence[Any],
    generator_metrics: Sequence[Any],
) -> None:
    """
    Example of wiring observed retriever and generator into a traced RAG
    pipeline that can be evaluated with DeepEval + Confident AI.
    """
    observed_retriever = make_observed_retriever(retriever_metrics)
    observed_generator = make_observed_generator(generator_metrics)

    def rag_pipeline(query: str) -> str:
        config = ObservedRetrieverConfig(retriever=retriever)
        ctx = observed_retriever(config, query)
        answer = observed_generator(generator_fn, query, ctx)
        return answer

    for golden in dataset.evals_iterator():
        rag_pipeline(golden.input)

