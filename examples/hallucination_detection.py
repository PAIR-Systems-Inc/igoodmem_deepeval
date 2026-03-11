"""
Hallucination Detection with GoodMem + DeepEval
================================================

Scenario: You're an engineer at a company that built a Q&A system powered by
GoodMem retrieval. Your product manager is worried the LLM might be making
things up. You need to PROVE with hard numbers that answers are grounded in
real documents — and catch any cases where the LLM hallucinates.

This script demonstrates how to:
1. Load your company knowledge base into GoodMem
2. Build two RAG pipelines (one faithful, one that hallucinates)
3. Use DeepEval's FaithfulnessMetric to catch the hallucinations
4. Show that hallucinating answers can still SEEM relevant (high relevancy
   score) while being unfaithful — proving why faithfulness testing matters

Usage:
    export GOODMEM_BASE_URL="http://localhost:8080"
    export GOODMEM_API_KEY="gm_..."
    export OPENAI_API_KEY="sk-..."
    python examples/hallucination_detection.py
"""
from __future__ import annotations

import os
import time

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

from goodmem_client.api.embedders_api import EmbeddersApi
from goodmem_client.api_client import ApiClient
from goodmem_client.configuration import Configuration

from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemRetriever,
    GoodMemRAGPipeline,
    compare_pipelines,
)

# ── Configuration ──────────────────────────────────────────────────────────

BASE_URL = os.environ.get("GOODMEM_BASE_URL", "http://localhost:8080")
API_KEY = os.environ.get("GOODMEM_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOODMEM_API_KEY before running this script.")

# ── Company knowledge base documents ──────────────────────────────────────
# These represent your real company docs that the Q&A system should answer from.

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

# Queries a customer might ask
TEST_QUERIES = [
    "What are the pricing plans for DataFlow Pro?",
    "What security certifications does DataFlow Pro have?",
    "What databases can DataFlow Pro connect to?",
]


# ── Generators ─────────────────────────────────────────────────────────────


def faithful_generator(query: str, context: list[str]) -> str:
    """
    A generator that ONLY uses information from the retrieved context.
    This simulates a well-behaved LLM that stays grounded.
    """
    if not context:
        return "I don't have enough information to answer that question."

    # Simply summarize what's in the context
    combined = " ".join(context[:2])

    if "pricing" in query.lower():
        return (
            "Based on our documentation: DataFlow Pro Standard is $499/month "
            "with up to 50,000 events/second and 5 pipeline slots. Enterprise "
            "is $1,499/month with unlimited events and pipelines. All plans "
            "include a 14-day free trial with no setup fees."
        )
    elif "security" in query.lower():
        return (
            "DataFlow Pro encrypts data at rest with AES-256 and in transit "
            "with TLS 1.3. We are SOC 2 Type II certified and GDPR compliant. "
            "Customer data is stored in US-East by default, with EU hosting "
            "available for Enterprise customers."
        )
    elif "database" in query.lower() or "connect" in query.lower():
        return (
            "DataFlow Pro integrates with PostgreSQL, MySQL, MongoDB, "
            "Snowflake, BigQuery, and Redshift as destinations. Source "
            "connectors include Kafka, Kinesis, Pulsar, REST APIs, and S3."
        )

    return f"Based on our docs: {combined[:200]}"


def hallucinating_generator(query: str, context: list[str]) -> str:
    """
    A generator that adds plausible-sounding but MADE UP information
    not found anywhere in the retrieved context. This simulates an LLM
    that confidently fabricates facts.
    """
    if "pricing" in query.lower():
        return (
            "DataFlow Pro Standard is $499/month. Enterprise is $1,499/month. "
            # ↓↓↓ HALLUCINATED: None of this is in the docs ↓↓↓
            "We also offer a Startup plan at $99/month for companies with "
            "fewer than 10 employees. Annual billing gives you a 30% discount. "
            "We recently partnered with AWS to offer $5,000 in free credits "
            "to new Enterprise customers."
        )
    elif "security" in query.lower():
        return (
            "DataFlow Pro uses AES-256 encryption and TLS 1.3. We are SOC 2 "
            "Type II certified. "
            # ↓↓↓ HALLUCINATED: Made up certifications and features ↓↓↓
            "We are also ISO 27001 certified and FedRAMP authorized for "
            "government use. Our zero-knowledge architecture means even our "
            "own engineers cannot access your data. We completed our HIPAA "
            "certification last quarter."
        )
    elif "database" in query.lower() or "connect" in query.lower():
        return (
            "DataFlow Pro connects to PostgreSQL, MySQL, MongoDB, Snowflake, "
            "BigQuery, and Redshift. "
            # ↓↓↓ HALLUCINATED: Made up integrations ↓↓↓
            "We also just launched native connectors for Oracle, Cassandra, "
            "and Elasticsearch. Our partnership with Databricks provides "
            "zero-config integration with Delta Lake. Real-time CDC from "
            "any database is included free on all plans."
        )

    return "I can help with that, but I'll need to make some assumptions."


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("HALLUCINATION DETECTION — GoodMem + DeepEval")
    print("=" * 60)

    # Step 1: Connect to GoodMem
    print("\n📡 Connecting to GoodMem...")
    client = GoodMemEvalClient(base_url=BASE_URL, api_key=API_KEY)

    # Find an embedder
    cfg = Configuration()
    cfg.host = BASE_URL
    cfg.api_key = {"ApiKeyAuth": API_KEY}
    embedder_client = ApiClient(configuration=cfg)
    embedders = EmbeddersApi(embedder_client).list_embedders().embedders
    if not embedders:
        raise RuntimeError("No embedders configured. Set one up in GoodMem first.")
    embedder_id = embedders[0].embedder_id
    print(f"   Using embedder: {embedder_id}")

    # Step 2: Create space and load company docs
    spaces = client.list_spaces()
    existing = next((s for s in spaces if s.name == "company-kb"), None)
    if existing:
        space_id = existing.space_id
        print(f"\n📂 Reusing existing space: {space_id}")
    else:
        space = client.create_space(
            space_name="company-kb",
            embedder=embedder_id,
            chunk_size=512,
            chunk_overlap=50,
        )
        space_id = space.space_id
        print(f"\n📂 Created space: {space_id}")

        print("📝 Loading company knowledge base...")
        for doc in COMPANY_DOCS:
            client.create_memory(
                space=space_id,
                text_content=doc["text"],
                source=doc["source"],
                tags=doc["tags"],
                author="knowledge-base",
            )
            print(f"   ✓ Loaded: {doc['source']}")

        print("⏳ Waiting for indexing...")
        time.sleep(3)

    # Step 3: Build the two pipelines
    print("\n🔧 Building pipelines...")
    retriever = GoodMemRetriever(client=client, spaces=[space_id], maximum_results=3)

    faithful_pipeline = GoodMemRAGPipeline(
        retriever=retriever, generator=faithful_generator
    )
    hallucinating_pipeline = GoodMemRAGPipeline(
        retriever=retriever, generator=hallucinating_generator
    )
    print("   ✓ Faithful pipeline (answers only from retrieved docs)")
    print("   ✓ Hallucinating pipeline (adds made-up facts)")

    # Step 4: Run evaluation
    print("\n🧪 Running DeepEval evaluation...")
    print("   This uses GPT as a judge to score each answer.\n")

    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
    ]

    results = compare_pipelines(
        queries=TEST_QUERIES,
        pipelines={
            "faithful": faithful_pipeline,
            "hallucinating": hallucinating_pipeline,
        },
        metrics=metrics,
    )

    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for system_name, result in results.items():
        print(f"\n{'🟢' if system_name == 'faithful' else '🔴'} {system_name.upper()} PIPELINE")
        print(f"   {result}")

    print("\n" + "=" * 60)
    print("WHAT THIS MEANS")
    print("=" * 60)
    print("""
The FAITHFUL pipeline should score high on both Faithfulness and Relevancy
because it only uses information that appears in the retrieved documents.

The HALLUCINATING pipeline may score high on Answer Relevancy (the answers
SOUND relevant to the question) but LOW on Faithfulness (the answers contain
claims not supported by the retrieved context).

This is exactly why Faithfulness testing matters — a hallucinating system
can produce answers that seem helpful but contain fabricated information.
By running FaithfulnessMetric regularly, you can catch these issues before
they reach your users.
""")


if __name__ == "__main__":
    main()
