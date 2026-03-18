#!/usr/bin/env python3
"""
Scaled Multi-Provider RAG Benchmark
====================================

Compares retrieval quality across multiple vector search providers using
real-world data (SQuAD 2.0) and DeepEval metrics.

Usage:
    # Quick smoke test (5 articles, 10 queries)
    python examples/benchmark/run_benchmark.py --num-articles 5 --max-queries 10

    # Medium run
    python examples/benchmark/run_benchmark.py --num-articles 50 --top-k 1,3

    # Full benchmark (~200 articles, ~400 queries)
    python examples/benchmark/run_benchmark.py --num-articles 200

    # Export results to JSON
    python examples/benchmark/run_benchmark.py --num-articles 50 --output results.json

Environment Variables:
    GOODMEM_BASE_URL    GoodMem server URL (default: http://localhost:8080)
    GOODMEM_API_KEY     GoodMem API key (required)
    OPENAI_API_KEY      OpenAI API key (required for generation + DeepEval judging)
    VECTARA_API_KEY     Vectara API key (required if --providers includes vectara)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .datasets import load_squad_subset
from .providers import GoodMemProvider, VectaraProvider
from .providers.base import RetrievalProvider
from .runner import (
    export_results_json,
    make_openai_generator,
    print_apples_to_apples,
    print_summary_table,
    run_benchmark,
)

PROVIDER_REGISTRY = {
    "goodmem": GoodMemProvider,
    "vectara": VectaraProvider,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaled multi-provider RAG benchmark using SQuAD 2.0 + DeepEval"
    )
    parser.add_argument(
        "--num-articles",
        type=int,
        default=200,
        help="Number of SQuAD articles to use (default: 200)",
    )
    parser.add_argument(
        "--queries-per-article",
        type=int,
        default=2,
        help="Q&A pairs per article (default: 2)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit total queries evaluated (useful for quick runs)",
    )
    parser.add_argument(
        "--top-k",
        type=str,
        default="1,3",
        help="Comma-separated top-k values to test (default: 1,3)",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="goodmem,vectara",
        help=f"Comma-separated providers to benchmark (available: {', '.join(PROVIDER_REGISTRY)})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Export results to JSON file at this path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset selection (default: 42)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for answer generation (default: gpt-4o-mini)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate env vars
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    # Parse config
    top_k_values = [int(k.strip()) for k in args.top_k.split(",")]
    provider_names = [p.strip().lower() for p in args.providers.split(",")]

    for name in provider_names:
        if name not in PROVIDER_REGISTRY:
            print(
                f"ERROR: Unknown provider '{name}'. "
                f"Available: {', '.join(PROVIDER_REGISTRY)}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Print banner
    print("=" * 70)
    print("SCALED MULTI-PROVIDER RAG BENCHMARK")
    print("=" * 70)
    print(f"  Providers:    {', '.join(provider_names)}")
    print(f"  Top-k values: {top_k_values}")
    print(f"  Articles:     {args.num_articles}")
    print(f"  Queries/art:  {args.queries_per_article}")
    if args.max_queries:
        print(f"  Max queries:  {args.max_queries}")
    print(f"  Generator:    {args.model}")
    print(f"  Seed:         {args.seed}")
    print()

    # Load dataset
    print("── Loading Dataset ──")
    dataset = load_squad_subset(
        num_articles=args.num_articles,
        queries_per_article=args.queries_per_article,
        seed=args.seed,
    )

    # Create providers
    providers: list[RetrievalProvider] = []
    for name in provider_names:
        providers.append(PROVIDER_REGISTRY[name]())

    # Create generator
    generator = make_openai_generator(api_key=openai_key, model=args.model)

    # Run benchmark
    start_time = time.time()
    results = run_benchmark(
        dataset=dataset,
        providers=providers,
        top_k_values=top_k_values,
        generator=generator,
        max_queries=args.max_queries,
    )
    total_time = time.time() - start_time

    # Report results
    print_summary_table(results)
    print_apples_to_apples(results)

    print(f"\n⏱️  Total benchmark time: {total_time:.1f}s")

    if args.output:
        export_results_json(results, args.output)

    # Teardown
    for provider in providers:
        provider.teardown()

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
