"""
Real LLM RAG Evaluation with GoodMem + OpenAI + DeepEval
=========================================================

This is the "real deal" example — it uses an actual LLM (OpenAI GPT) to
generate answers from GoodMem-retrieved context, then evaluates everything
with DeepEval.

Unlike the other examples that use hardcoded generators, this one calls
the OpenAI API for every query, making it a true end-to-end RAG evaluation.

Usage:
    export GOODMEM_BASE_URL="http://localhost:8080"
    export GOODMEM_API_KEY="gm_..."
    export OPENAI_API_KEY="sk-..."
    python examples/openai_rag_eval.py
"""
from __future__ import annotations

import os
import time

from openai import OpenAI

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)

from goodmem_client.api.embedders_api import EmbeddersApi
from goodmem_client.api_client import ApiClient
from goodmem_client.configuration import Configuration

from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemRetriever,
    GoodMemRAGPipeline,
    evaluate_goodmem_rag,
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
    """
    Call OpenAI GPT to generate an answer grounded in the retrieved context.
    This is what a real RAG system does.
    """
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


# ── Knowledge base documents ──────────────────────────────────────────────

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


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("REAL LLM RAG EVALUATION — GoodMem + OpenAI + DeepEval")
    print("=" * 60)

    # Step 1: Connect and set up
    print("\n📡 Connecting to GoodMem...")
    client = GoodMemEvalClient(base_url=BASE_URL, api_key=API_KEY)

    cfg = Configuration()
    cfg.host = BASE_URL
    cfg.api_key = {"ApiKeyAuth": API_KEY}
    embedder_client = ApiClient(configuration=cfg)
    embedders = EmbeddersApi(embedder_client).list_embedders().embedders
    if not embedders:
        raise RuntimeError("No embedders configured. Set one up in GoodMem first.")
    embedder_id = embedders[0].embedder_id

    # Step 2: Create or reuse space
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

    # Step 3: Build the real RAG pipeline
    print("\n🔧 Building pipeline: GoodMem retriever + OpenAI GPT-4o-mini generator")
    retriever = GoodMemRetriever(client=client, spaces=[space_id], maximum_results=3)
    pipeline = GoodMemRAGPipeline(retriever=retriever, generator=openai_generator)

    # Step 4: Run evaluation
    print("\n🧪 Running evaluation (this calls OpenAI for each query)...\n")

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.5),
    ]

    result = evaluate_goodmem_rag(
        queries=TEST_QUERIES,
        pipeline=pipeline,
        metrics=metrics,
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(result)
    print("""
This evaluation used:
  - GoodMem for semantic retrieval (finding relevant docs)
  - OpenAI GPT-4o-mini for answer generation (real LLM, not canned strings)
  - DeepEval GPT-4.1 as the judge (scoring relevancy, faithfulness, context)

This is a true end-to-end RAG evaluation.
""")


if __name__ == "__main__":
    main()
