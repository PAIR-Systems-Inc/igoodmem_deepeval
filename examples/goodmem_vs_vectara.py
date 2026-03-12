"""
GoodMem vs Vectara — Head-to-Head RAG Comparison with DeepEval
================================================================

This example evaluates GoodMem and Vectara side-by-side on the same
documents and queries using DeepEval metrics. Both systems get the
exact same knowledge base and test questions, and both use the same
OpenAI GPT-4o-mini generator — so the only variable is retrieval quality.

Prerequisites:
    pip install vectara-client

Usage:
    export GOODMEM_BASE_URL="http://localhost:8080"
    export GOODMEM_API_KEY="gm_..."
    export OPENAI_API_KEY="sk-..."
    export VECTARA_API_KEY="..."
    python examples/goodmem_vs_vectara.py
"""
from __future__ import annotations

import os
import time
from typing import List, Tuple

from openai import OpenAI

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)

from goodmem_client.api.embedders_api import EmbeddersApi
from goodmem_client.api_client import ApiClient as GoodMemApiClient
from goodmem_client.configuration import Configuration as GoodMemConfiguration

from vectara_client import (
    ApiClient as VectaraApiClient,
    Configuration as VectaraConfiguration,
    CorporaApi,
    IndexApi,
    QueriesApi,
    CreateCorpusRequest,
    CoreDocument,
    CoreDocumentPart,
    QueryCorpusRequest,
    SearchCorpusParameters,
)

from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemRetriever,
    GoodMemRAGPipeline,
    compare_pipelines,
)

# ── Configuration ──────────────────────────────────────────────────────────

GOODMEM_BASE_URL = os.environ.get("GOODMEM_BASE_URL", "http://localhost:8080")
GOODMEM_API_KEY = os.environ.get("GOODMEM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
VECTARA_API_KEY = os.environ.get("VECTARA_API_KEY")

for name, val in [
    ("GOODMEM_API_KEY", GOODMEM_API_KEY),
    ("OPENAI_API_KEY", OPENAI_API_KEY),
    ("VECTARA_API_KEY", VECTARA_API_KEY),
]:
    if not val:
        raise RuntimeError(f"Set {name} before running this script.")

# ── Shared OpenAI generator ───────────────────────────────────────────────

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def openai_generator(query: str, context: list[str]) -> str:
    """Same generator for both pipelines — keeps the comparison fair."""
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


# ── Shared knowledge base ─────────────────────────────────────────────────

COMPANY_DOCS = [
    {
        "id": "product-overview",
        "text": (
            "DataFlow Pro is our flagship data pipeline product. It supports "
            "real-time streaming from Kafka, Kinesis, and Pulsar sources. "
            "Batch processing is available via scheduled jobs with a minimum "
            "interval of 15 minutes. The product handles up to 50,000 events "
            "per second on the Standard plan."
        ),
        "source": "product-docs",
    },
    {
        "id": "pricing-page",
        "text": (
            "Pricing: DataFlow Pro Standard is $499/month and includes up to "
            "50,000 events/second, 5 pipeline slots, and email support. "
            "DataFlow Pro Enterprise is $1,499/month and includes unlimited "
            "events, unlimited pipelines, dedicated support, and SSO. "
            "All plans include a 14-day free trial. No setup fees."
        ),
        "source": "pricing-page",
    },
    {
        "id": "support-policy",
        "text": (
            "Support hours: Our support team is available Monday through Friday, "
            "9 AM to 6 PM Eastern Time. Enterprise customers receive 24/7 "
            "priority support with a guaranteed 1-hour response time. Standard "
            "plan customers receive a 24-hour response time SLA. Weekend "
            "support is only available for Severity 1 incidents on Enterprise plans."
        ),
        "source": "support-policy",
    },
    {
        "id": "security-docs",
        "text": (
            "Security: DataFlow Pro encrypts all data at rest using AES-256 "
            "and in transit using TLS 1.3. We are SOC 2 Type II certified "
            "and GDPR compliant. Customer data is stored in the US-East region "
            "by default, with EU hosting available for Enterprise customers. "
            "We do not sell or share customer data with third parties."
        ),
        "source": "security-docs",
    },
    {
        "id": "integration-docs",
        "text": (
            "Integrations: DataFlow Pro natively integrates with PostgreSQL, "
            "MySQL, MongoDB, Snowflake, BigQuery, and Redshift as destinations. "
            "Source connectors are available for Kafka, Kinesis, Pulsar, "
            "REST APIs, and S3. Custom connectors can be built using our "
            "Connector SDK (Python and Java supported)."
        ),
        "source": "integration-docs",
    },
]

TEST_QUERIES = [
    "What are the pricing plans for DataFlow Pro?",
    "What security certifications does DataFlow Pro have?",
    "What databases can DataFlow Pro connect to?",
    "What are the support hours?",
]


# ── Vectara pipeline wrapper ──────────────────────────────────────────────


class VectaraRAGPipeline:
    """Wraps Vectara retrieval + shared generator into a RAGPipeline callable."""

    def __init__(
        self,
        queries_api: QueriesApi,
        corpus_key: str,
        generator,
        max_results: int = 3,
    ):
        self._queries_api = queries_api
        self._corpus_key = corpus_key
        self._generator = generator
        self._max_results = max_results

    def __call__(self, query: str) -> Tuple[str, List[str]]:
        request = QueryCorpusRequest(
            query=query,
            search=SearchCorpusParameters(
                limit=self._max_results,
            ),
        )
        response = self._queries_api.query_corpus(
            corpus_key=self._corpus_key,
            query_corpus_request=request,
        )

        retrieval_context = []
        if response.search_results:
            for result in response.search_results[: self._max_results]:
                if result.text:
                    retrieval_context.append(result.text)

        answer = self._generator(query, retrieval_context)
        return answer, retrieval_context


# ── Setup helpers ──────────────────────────────────────────────────────────


def setup_goodmem(max_results: int = 3):
    """Set up GoodMem space and return a pipeline."""
    print("\n── Setting up GoodMem ──")
    client = GoodMemEvalClient(base_url=GOODMEM_BASE_URL, api_key=GOODMEM_API_KEY)

    cfg = GoodMemConfiguration()
    cfg.host = GOODMEM_BASE_URL
    cfg.api_key = {"ApiKeyAuth": GOODMEM_API_KEY}
    gm_api_client = GoodMemApiClient(configuration=cfg)

    embedders = EmbeddersApi(gm_api_client).list_embedders().embedders
    if not embedders:
        raise RuntimeError("No embedders configured in GoodMem.")
    embedder_id = embedders[0].embedder_id

    spaces = client.list_spaces()
    existing = next((s for s in spaces if s.name == "company-kb"), None)
    if existing:
        space_id = existing.space_id
        print(f"  📂 Reusing existing space: {space_id}")
    else:
        space = client.create_space(
            space_name="company-kb",
            embedder=embedder_id,
            chunk_size=512,
            chunk_overlap=50,
        )
        space_id = space.space_id
        print(f"  📂 Created space: {space_id}")

        for doc in COMPANY_DOCS:
            client.create_memory(
                space=space_id,
                text_content=doc["text"],
                source=doc["source"],
                author="knowledge-base",
            )
        print(f"  📝 Loaded {len(COMPANY_DOCS)} documents")
        print("  ⏳ Waiting for indexing...")
        time.sleep(3)

    retriever = GoodMemRetriever(
        client=client, spaces=[space_id], maximum_results=max_results,
    )
    pipeline = GoodMemRAGPipeline(retriever=retriever, generator=openai_generator)
    print("  ✅ GoodMem pipeline ready")
    return pipeline


def setup_vectara(max_results: int = 3):
    """Set up Vectara corpus and return a pipeline."""
    print("\n── Setting up Vectara ──")

    vcfg = VectaraConfiguration()
    vcfg.api_key["ApiKeyAuth"] = VECTARA_API_KEY
    vectara_api_client = VectaraApiClient(configuration=vcfg)

    corpora_api = CorporaApi(vectara_api_client)
    index_api = IndexApi(vectara_api_client)
    queries_api = QueriesApi(vectara_api_client)

    corpus_key = "goodmem-vs-vectara-eval"

    # Try to reuse existing corpus, create if not found
    try:
        corpus = corpora_api.get_corpus(corpus_key=corpus_key)
        print(f"  📂 Reusing existing corpus: {corpus_key}")
    except Exception:
        corpus = corpora_api.create_corpus(
            create_corpus_request=CreateCorpusRequest(
                key=corpus_key,
                name="GoodMem vs Vectara Eval",
                description="Comparison corpus for DeepEval benchmarking",
            )
        )
        print(f"  📂 Created corpus: {corpus_key}")

        # Ingest same documents
        for doc in COMPANY_DOCS:
            core_doc = CoreDocument(
                id=doc["id"],
                type="core",
                metadata={"source": doc["source"]},
                document_parts=[
                    CoreDocumentPart(text=doc["text"]),
                ],
            )
            index_api.create_corpus_document(
                corpus_key=corpus_key,
                request_body=core_doc,
            )
        print(f"  📝 Loaded {len(COMPANY_DOCS)} documents")
        print("  ⏳ Waiting for indexing...")
        time.sleep(5)

    pipeline = VectaraRAGPipeline(
        queries_api=queries_api,
        corpus_key=corpus_key,
        generator=openai_generator,
        max_results=max_results,
    )
    print("  ✅ Vectara pipeline ready")
    return pipeline


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("GOODMEM vs VECTARA — Head-to-Head RAG Evaluation")
    print("=" * 70)
    print()
    print("Same docs, same queries, same generator — only retrieval differs.")

    # Set up both pipelines
    goodmem_pipeline = setup_goodmem(max_results=3)
    vectara_pipeline = setup_vectara(max_results=3)

    # Run comparison
    print("\n🧪 Running evaluation (this calls OpenAI for each query per pipeline)...\n")

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.5),
    ]

    results = compare_pipelines(
        queries=TEST_QUERIES,
        pipelines={
            "GoodMem": goodmem_pipeline,
            "Vectara": vectara_pipeline,
        },
        metrics=metrics,
    )

    # Print results
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    for pipeline_name, pipeline_results in results.items():
        print(f"\n📊 {pipeline_name}:")
        if "error" in pipeline_results:
            print(f"   ❌ Error: {pipeline_results['error']}")
            continue
        for tr in pipeline_results.get("test_results", []):
            print(f"   Query: {tr.input[:50]}...")
            for md in tr.metrics_data:
                status = "✅" if md.success else "❌"
                print(f"     {status} {md.name}: {md.score:.2f}")

    print()
    print("Both pipelines used:")
    print("  - Same 5 DataFlow Pro documents")
    print("  - Same 4 test queries")
    print("  - Same OpenAI GPT-4o-mini generator")
    print("  - DeepEval GPT-4.1 as the judge")
    print()
    print("The only difference is the retrieval engine (GoodMem vs Vectara).")
    print()


if __name__ == "__main__":
    main()
