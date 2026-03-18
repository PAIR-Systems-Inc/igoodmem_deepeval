from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from .pipeline import RAGPipeline


def compare_pipelines(
    queries: Sequence[str],
    pipelines: Mapping[str, RAGPipeline],
    metrics: Sequence[Any],
    expected_outputs: Optional[Sequence[Optional[str]]] = None,
) -> Dict[str, Any]:
    """
    Run the same set of queries through multiple RAG pipelines (e.g. GoodMem
    vs another retriever) and return evaluation results per system.

    Each pipeline must accept (query: str) and return (answer: str,
    retrieval_context: list[str]).

    Args:
        queries: The evaluation queries.
        pipelines: Mapping of system name to RAGPipeline callable.
        metrics: DeepEval metrics to evaluate with.
        expected_outputs: Optional gold-standard answers (one per query).
            When provided, populates LLMTestCase.expected_output for metrics
            like CorrectnessMetric.
    """
    results: Dict[str, Any] = {}
    for system_name, pipeline in pipelines.items():
        test_cases: List[LLMTestCase] = []
        for i, q in enumerate(queries):
            answer, ctx = pipeline(q)
            tc = LLMTestCase(
                input=q,
                actual_output=answer,
                retrieval_context=ctx,
            )
            if expected_outputs and i < len(expected_outputs) and expected_outputs[i]:
                tc.expected_output = expected_outputs[i]
            test_cases.append(tc)
        results[system_name] = evaluate(test_cases=test_cases, metrics=list(metrics))
    return results

