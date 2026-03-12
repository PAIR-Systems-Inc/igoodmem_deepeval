"""
Retrieval Tuning Evaluation — How Retrieval Settings Affect RAG Quality
========================================================================

This example shows how tuning GoodMem retrieval parameters impacts evaluation
scores. It runs the same queries with different configurations and prints a
side-by-side comparison.

Configurations compared:
  1. BROAD retrieval  — maximum_results=3 (returns more chunks, but many irrelevant)
  2. PRECISE retrieval — maximum_results=1 (returns only the top chunk)

This demonstrates how to use DeepEval metrics to find the right balance
between recall (getting enough context) and precision (avoiding noise).

Usage:
    export GOODMEM_BASE_URL="http://localhost:8080"
    export GOODMEM_API_KEY="gm_..."
    export OPENAI_API_KEY="sk-..."
    python examples/retrieval_tuning_eval.py
"""
from __future__ import annotations

import os
import time

from openai import OpenAI

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from goodmem_client.api.embedders_api import EmbeddersApi
from goodmem_client.api_client import ApiClient
from goodmem_client.configuration import Configuration

from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemRetriever,
    GoodMemRAGPipeline,
)

# ── Configuration ──────────────────────────────────────────────────────────

BASE_URL = os.environ.get("GOODMEM_BASE_URL", "http://localhost:8080")
API_KEY = os.environ.get("GOODMEM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("Set GOODMEM_API_KEY before running this script.")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY before running this script.")

# ── OpenAI generator ──────────────────────────────────────────────────────

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def openai_generator(query: str, context: list[str]) -> str:
    """Call OpenAI GPT to generate an answer grounded in the retrieved context."""
    context_text = "\n\n".join(context) if context else "No context available."
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
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


# ── Knowledge base ────────────────────────────────────────────────────────

COMPANY_DOCS = [
    {
        "text": (
            "DataFlow Pro is our flagship data pipeline product. It supports "
            "real-time streaming from Kafka, Kinesis, and Pulsar sources. "
            "Batch processing is available via scheduled jobs with a minimum "
            "interval of 15 minutes. The product handles up to 50,000 events "
            "per second on the Standard plan."
        ),
        "source": "product-docs",
        "tags": "product,streaming,batch",
    },
    {
        "text": (
            "Pricing: DataFlow Pro Standard is $499/month and includes up to "
            "50,000 events/second, 5 pipeline slots, and email support. "
            "DataFlow Pro Enterprise is $1,499/month and includes unlimited "
            "events, unlimited pipelines, dedicated support, and SSO. "
            "All plans include a 14-day free trial. No setup fees."
        ),
        "source": "pricing-page",
        "tags": "pricing,plans",
    },
    {
        "text": (
            "Support hours: Our support team is available Monday through Friday, "
            "9 AM to 6 PM Eastern Time. Enterprise customers receive 24/7 "
            "priority support with a guaranteed 1-hour response time. Standard "
            "plan customers receive a 24-hour response time SLA. Weekend "
            "support is only available for Severity 1 incidents on Enterprise plans."
        ),
        "source": "support-policy",
        "tags": "support,sla",
    },
    {
        "text": (
            "Security: DataFlow Pro encrypts all data at rest using AES-256 "
            "and in transit using TLS 1.3. We are SOC 2 Type II certified "
            "and GDPR compliant. Customer data is stored in the US-East region "
            "by default, with EU hosting available for Enterprise customers. "
            "We do not sell or share customer data with third parties."
        ),
        "source": "security-docs",
        "tags": "security,compliance",
    },
    {
        "text": (
            "Integrations: DataFlow Pro natively integrates with PostgreSQL, "
            "MySQL, MongoDB, Snowflake, BigQuery, and Redshift as destinations. "
            "Source connectors are available for Kafka, Kinesis, Pulsar, "
            "REST APIs, and S3. Custom connectors can be built using our "
            "Connector SDK (Python and Java supported)."
        ),
        "source": "integration-docs",
        "tags": "integrations,connectors",
    },
]

TEST_QUERIES = [
    "What are the pricing plans for DataFlow Pro?",
    "What security certifications does DataFlow Pro have?",
    "What databases can DataFlow Pro connect to?",
    "What are the support hours?",
]


# ── Helpers ────────────────────────────────────────────────────────────────


def run_pipeline_eval(pipeline: GoodMemRAGPipeline, label: str):
    """Run all queries through a pipeline and return per-query scores."""
    results = {}
    test_cases = []

    for query in TEST_QUERIES:
        answer, retrieval_context = pipeline(query)
        tc = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=retrieval_context,
        )
        test_cases.append(tc)

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.5),
    ]

    print(f"\n📊 Evaluating: {label}")
    eval_result = evaluate(test_cases=test_cases, metrics=metrics)

    for i, tr in enumerate(eval_result.test_results):
        scores = {}
        for md in tr.metrics_data:
            scores[md.name] = md.score
        results[TEST_QUERIES[i]] = scores

    return results


def print_comparison(broad_scores, precise_scores):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 95)
    print("BROAD (top-3) vs PRECISE (top-1) RETRIEVAL — Side-by-Side Comparison")
    print("=" * 95)

    header = f"{'Query':<45} {'Metric':<25} {'Top-3':>8} {'Top-1':>8} {'Delta':>8}"
    print(header)
    print("-" * 95)

    for query in TEST_QUERIES:
        short_q = query[:42] + "..." if len(query) > 45 else query
        b_scores = broad_scores.get(query, {})
        p_scores = precise_scores.get(query, {})

        for metric in ["Answer Relevancy", "Faithfulness", "Contextual Relevancy"]:
            b = b_scores.get(metric, 0)
            p = p_scores.get(metric, 0)
            delta = p - b
            sign = "+" if delta > 0 else ""
            q_col = short_q if metric == "Answer Relevancy" else ""
            print(f"{q_col:<45} {metric:<25} {b:>8.2f} {p:>8.2f} {sign}{delta:>7.2f}")
        print()

    print("💡 KEY TAKEAWAY:")
    print("   Top-1 retrieval boosts Contextual Relevancy by eliminating noise,")
    print("   while Answer Relevancy and Faithfulness remain high.")
    print("   Use DeepEval metrics to find your optimal maximum_results setting.")
    print()


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("RETRIEVAL TUNING EVAL — GoodMem + OpenAI + DeepEval")
    print("=" * 70)

    # Connect
    print("\n📡 Connecting to GoodMem...")
    client = GoodMemEvalClient(base_url=BASE_URL, api_key=API_KEY)

    cfg = Configuration()
    cfg.host = BASE_URL
    cfg.api_key = {"ApiKeyAuth": API_KEY}
    api_client = ApiClient(configuration=cfg)

    # Get embedder
    embedders = EmbeddersApi(api_client).list_embedders().embedders
    if not embedders:
        raise RuntimeError("No embedders configured in GoodMem.")
    embedder_id = embedders[0].embedder_id

    # Create or reuse space
    spaces = client.list_spaces()
    existing = next((s for s in spaces if s.name == "company-kb"), None)
    if existing:
        space_id = existing.space_id
        print(f"📂 Reusing existing space: {space_id}")
    else:
        space = client.create_space(
            space_name="company-kb",
            embedder=embedder_id,
            chunk_size=512,
            chunk_overlap=50,
        )
        space_id = space.space_id
        print(f"📂 Created space: {space_id}")

        print("📝 Loading knowledge base...")
        for doc in COMPANY_DOCS:
            client.create_memory(
                space=space_id,
                text_content=doc["text"],
                source=doc["source"],
                tags=doc["tags"],
                author="knowledge-base",
            )
        print(f"   Loaded {len(COMPANY_DOCS)} documents")
        print("⏳ Waiting for indexing...")
        time.sleep(3)

    # ── BROAD retrieval (top-3) ──
    print("\n🔧 Building pipeline: BROAD retrieval (maximum_results=3)...")
    retriever_broad = GoodMemRetriever(
        client=client, spaces=[space_id], maximum_results=3,
    )
    pipeline_broad = GoodMemRAGPipeline(
        retriever=retriever_broad, generator=openai_generator,
    )

    # ── PRECISE retrieval (top-1) ──
    print("🔧 Building pipeline: PRECISE retrieval (maximum_results=1)...")
    retriever_precise = GoodMemRetriever(
        client=client, spaces=[space_id], maximum_results=1,
    )
    pipeline_precise = GoodMemRAGPipeline(
        retriever=retriever_precise, generator=openai_generator,
    )

    # ── Run evaluations ──
    print("\n🧪 Running evaluations (this calls OpenAI for each query, twice)...\n")

    broad_scores = run_pipeline_eval(pipeline_broad, "BROAD retrieval (top-3)")
    precise_scores = run_pipeline_eval(pipeline_precise, "PRECISE retrieval (top-1)")

    # ── Print comparison ──
    print_comparison(broad_scores, precise_scores)


if __name__ == "__main__":
    main()
