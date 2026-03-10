from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from .pipeline import RAGPipeline


def compare_pipelines(
    queries: Sequence[str],
    pipelines: Mapping[str, RAGPipeline],
    metrics: Sequence[Any],
) -> Dict[str, Any]:
    """
    Run the same set of queries through multiple RAG pipelines (e.g. GoodMem
    vs another retriever) and return evaluation results per system.

    Each pipeline must accept (query: str) and return (answer: str,
    retrieval_context: list[str]).
    """
    results: Dict[str, Any] = {}
    for system_name, pipeline in pipelines.items():
        test_cases: List[LLMTestCase] = []
        for q in queries:
            answer, ctx = pipeline(q)
            test_cases.append(
                LLMTestCase(
                    input=q,
                    actual_output=answer,
                    retrieval_context=ctx,
                )
            )
        results[system_name] = evaluate(test_cases=test_cases, metrics=list(metrics))
    return results

