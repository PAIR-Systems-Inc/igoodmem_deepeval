"""
Benchmark runner — orchestrates dataset loading, provider setup, evaluation,
and result reporting.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Increase DeepEval's per-test-case timeout for long documents
os.environ.setdefault("DEEPEVAL_TIMEOUT", "600")

from openai import OpenAI

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from .datasets import BenchmarkDataset, BenchmarkQuery
from .providers.base import RAGPipeline, RetrievalProvider


# ── Shared generator ─────────────────────────────────────────────────────────


def make_openai_generator(api_key: str, model: str = "gpt-4o-mini"):
    """Create a generator function using OpenAI."""
    client = OpenAI(api_key=api_key)

    def generator(query: str, context: Sequence[str]) -> str:
        context_text = "\n\n".join(context) if context else "No context available."
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question "
                        "using ONLY the information provided in the context below. "
                        "If the context doesn't contain enough information, say so. "
                        "Do not make up information.\n\n"
                        f"Context:\n{context_text}"
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content

    return generator


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class QueryScore:
    query: str
    expected_answer: Optional[str]
    actual_output: str
    retrieval_context: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConfigResult:
    provider_name: str
    top_k: int
    scores: List[QueryScore] = field(default_factory=list)
    eval_time_seconds: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.provider_name} (top-{self.top_k})"

    def avg_score(self, metric_name: str) -> float:
        vals = [s.metrics.get(metric_name, 0) for s in self.scores]
        return sum(vals) / len(vals) if vals else 0.0


# ── Core runner ──────────────────────────────────────────────────────────────


def _run_single_config(
    pipeline: RAGPipeline,
    queries: List[BenchmarkQuery],
    label: str,
) -> ConfigResult:
    """Run all queries through a pipeline and evaluate with DeepEval."""
    print(f"\n📊 Evaluating: {label}")

    test_cases: List[LLMTestCase] = []
    raw_outputs: List[tuple] = []

    for i, bq in enumerate(queries):
        answer, ctx = pipeline(bq.query)
        # Truncate long context chunks to avoid DeepEval/OpenAI timeouts.
        # The LLM judge doesn't need 20K+ char contexts to assess relevancy.
        ctx = [c[:1500] for c in ctx[:5]]
        raw_outputs.append((answer, ctx))

        tc = LLMTestCase(
            input=bq.query,
            actual_output=answer,
            retrieval_context=ctx,
        )
        if bq.expected_answer:
            tc.expected_output = bq.expected_answer
        test_cases.append(tc)

        # Progress every 50 queries
        if (i + 1) % 50 == 0:
            print(f"  ⏳ Retrieved {i + 1}/{len(queries)} queries...")

    print(f"  🧪 Running DeepEval metrics on {len(test_cases)} test cases...")

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.5),
    ]

    start = time.time()
    from deepeval.evaluate.configs import ErrorConfig
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        error_config=ErrorConfig(ignore_errors=True),
    )
    eval_time = time.time() - start

    # Build structured results
    config_result = ConfigResult(
        provider_name=label.split(" (")[0] if " (" in label else label,
        top_k=0,
        eval_time_seconds=eval_time,
    )

    for i, tr in enumerate(result.test_results):
        qs = QueryScore(
            query=queries[i].query,
            expected_answer=queries[i].expected_answer,
            actual_output=raw_outputs[i][0],
            retrieval_context=raw_outputs[i][1],
        )
        for md in tr.metrics_data:
            qs.metrics[md.name] = md.score if md.score is not None else 0.0
        config_result.scores.append(qs)

    return config_result


def run_benchmark(
    dataset: BenchmarkDataset,
    providers: List[RetrievalProvider],
    top_k_values: List[int],
    generator,
    max_queries: Optional[int] = None,
) -> List[ConfigResult]:
    """
    Run the full benchmark: index docs, evaluate each provider x top_k.

    Args:
        dataset: The BenchmarkDataset to evaluate against.
        providers: List of RetrievalProvider instances.
        top_k_values: List of top-k values to test (e.g. [1, 3, 5]).
        generator: The shared generator function.
        max_queries: If set, limit evaluation to this many queries.

    Returns:
        List of ConfigResult objects with per-query scores.
    """
    queries = dataset.queries
    if max_queries and max_queries < len(queries):
        queries = queries[:max_queries]
        print(f"  ℹ️  Limiting to {max_queries} queries (of {len(dataset.queries)})")

    # Setup all providers
    for provider in providers:
        print(f"\n── Setting up {provider.name} ──")
        provider.setup(dataset.documents)

    # Run evaluations
    all_results: List[ConfigResult] = []
    total_configs = len(providers) * len(top_k_values)
    config_num = 0

    for provider in providers:
        for top_k in top_k_values:
            config_num += 1
            label = f"{provider.name} (top-{top_k})"
            print(
                f"\n{'=' * 70}\n"
                f"Config {config_num}/{total_configs}: {label}\n"
                f"{'=' * 70}"
            )

            pipeline = provider.make_pipeline(
                generator=generator,
                top_k=top_k,
            )
            result = _run_single_config(
                pipeline=pipeline,
                queries=queries,
                label=label,
            )
            result.top_k = top_k
            all_results.append(result)

    return all_results


# ── Reporting ────────────────────────────────────────────────────────────────


METRIC_NAMES = ["Answer Relevancy", "Faithfulness", "Contextual Relevancy"]


def print_summary_table(results: List[ConfigResult]) -> None:
    """Print a compact summary table of average scores per config."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY — Average Scores")
    print("=" * 100)

    header = f"{'Configuration':<30}"
    for m in METRIC_NAMES:
        header += f" {m:>22}"
    header += f" {'Time':>10}"
    print(header)
    print("-" * 100)

    for r in results:
        row = f"{r.label:<30}"
        for m in METRIC_NAMES:
            row += f" {r.avg_score(m):>22.3f}"
        row += f" {r.eval_time_seconds:>8.1f}s"
        print(row)

    print("=" * 100)


def print_apples_to_apples(results: List[ConfigResult]) -> None:
    """Print head-to-head comparison for each top-k value."""
    # Group by top_k
    by_top_k: Dict[int, List[ConfigResult]] = {}
    for r in results:
        by_top_k.setdefault(r.top_k, []).append(r)

    print("\n" + "=" * 100)
    print("APPLES-TO-APPLES COMPARISON (Contextual Relevancy)")
    print("=" * 100)

    for top_k, configs in sorted(by_top_k.items()):
        if len(configs) < 2:
            continue
        print(f"\n  Top-{top_k}:")
        header = f"  {'Provider':<25}"
        header += f" {'Avg Score':>12} {'Pass Rate':>12}"
        print(header)
        print("  " + "-" * 55)

        for c in configs:
            avg = c.avg_score("Contextual Relevancy")
            passing = sum(
                1
                for s in c.scores
                if s.metrics.get("Contextual Relevancy", 0) >= 0.5
            )
            rate = passing / len(c.scores) * 100 if c.scores else 0
            print(f"  {c.provider_name:<25} {avg:>12.3f} {rate:>11.1f}%")


def export_results_json(results: List[ConfigResult], path: str) -> None:
    """Export full results to a JSON file."""
    output = []
    for r in results:
        config_data = {
            "provider": r.provider_name,
            "top_k": r.top_k,
            "label": r.label,
            "eval_time_seconds": r.eval_time_seconds,
            "averages": {m: r.avg_score(m) for m in METRIC_NAMES},
            "queries": [
                {
                    "query": s.query,
                    "expected_answer": s.expected_answer,
                    "actual_output": s.actual_output,
                    "retrieval_context": s.retrieval_context,
                    "metrics": s.metrics,
                }
                for s in r.scores
            ],
        }
        output.append(config_data)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n📁 Results exported to {path}")
