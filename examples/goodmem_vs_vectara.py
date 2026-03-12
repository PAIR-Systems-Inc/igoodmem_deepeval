"""
GoodMem vs Vectara — Fair Cross-System RAG Comparison with DeepEval
====================================================================

This example evaluates GoodMem and Vectara side-by-side on the same
documents and queries using DeepEval metrics. Both systems get the
exact same knowledge base and test questions, and both use the same
OpenAI GPT-4o-mini generator — so the only variable is retrieval.

Five configurations are compared (apples-to-apples where possible):
  1. GoodMem (top-3)               — default retrieval, 3 chunks
  2. Vectara (top-3)               — default retrieval, 3 chunks
  3. GoodMem (top-1)               — precise retrieval, 1 chunk
  4. Vectara (top-1)               — precise retrieval, 1 chunk
  5. GoodMem + metadata filter     — targeted retrieval using source metadata

Configs 1 vs 2 and 3 vs 4 are direct apples-to-apples comparisons.
Config 5 showcases GoodMem's metadata filtering capability.

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

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

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
    CreateDocumentRequest,
    CoreDocument,
    CoreDocumentPart,
    QueryCorpusRequest,
    SearchCorpusParameters,
)

from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemRetriever,
    GoodMemRAGPipeline,
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
    """Same generator for all pipelines — keeps the comparison fair."""
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

# Each query is paired with the source metadata filter for config 5.
TEST_QUERIES_WITH_FILTERS = [
    {
        "query": "What are the pricing plans for DataFlow Pro?",
        "filter": "CAST(val('$.source') AS TEXT) = 'pricing-page'",
    },
    {
        "query": "What security certifications does DataFlow Pro have?",
        "filter": "CAST(val('$.source') AS TEXT) = 'security-docs'",
    },
    {
        "query": "What databases can DataFlow Pro connect to?",
        "filter": "CAST(val('$.source') AS TEXT) = 'integration-docs'",
    },
    {
        "query": "What are the support hours?",
        "filter": "CAST(val('$.source') AS TEXT) = 'support-policy'",
    },
]

TEST_QUERIES = [q["query"] for q in TEST_QUERIES_WITH_FILTERS]


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


class GoodMemFilteredPipeline:
    """
    GoodMem pipeline that uses per-query metadata filters.
    Demonstrates metadata filtering — target specific document sources per query.
    """

    def __init__(self, client, space_id, generator, query_filters):
        self._client = client
        self._space_id = space_id
        self._generator = generator
        # Map query -> filter expression
        self._query_filters = {qf["query"]: qf["filter"] for qf in query_filters}

    def __call__(self, query: str) -> Tuple[str, List[str]]:
        metadata_filter = self._query_filters.get(query)
        retriever = GoodMemRetriever(
            client=self._client,
            spaces=[self._space_id],
            maximum_results=3,
            metadata_filter=metadata_filter,
        )
        chunks = retriever.retrieve(query)
        retrieval_context = retriever.to_text_list(chunks)
        answer = self._generator(query, retrieval_context)
        return answer, retrieval_context


# ── Setup helpers ──────────────────────────────────────────────────────────


def setup_goodmem():
    """Set up GoodMem space and return client + space_id."""
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

    print("  ✅ GoodMem ready")
    return client, space_id


def setup_vectara():
    """Set up Vectara corpus and return queries_api + corpus_key."""
    print("\n── Setting up Vectara ──")

    vcfg = VectaraConfiguration()
    vcfg.api_key["ApiKeyAuth"] = VECTARA_API_KEY
    vectara_api_client = VectaraApiClient(configuration=vcfg)

    corpora_api = CorporaApi(vectara_api_client)
    index_api = IndexApi(vectara_api_client)
    queries_api = QueriesApi(vectara_api_client)

    corpus_key = "goodmem-vs-vectara-eval"

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
                create_document_request=CreateDocumentRequest(
                    actual_instance=core_doc,
                ),
            )
        print(f"  📝 Loaded {len(COMPANY_DOCS)} documents")
        print("  ⏳ Waiting for indexing...")
        time.sleep(5)

    print("  ✅ Vectara ready")
    return queries_api, corpus_key


# ── Evaluation helper ──────────────────────────────────────────────────────


def run_eval(pipeline, label):
    """Run queries through a pipeline and return per-query scores."""
    test_cases = []
    for query in TEST_QUERIES:
        answer, ctx = pipeline(query)
        test_cases.append(LLMTestCase(
            input=query, actual_output=answer, retrieval_context=ctx,
        ))

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.5),
    ]

    print(f"\n📊 Evaluating: {label}")
    result = evaluate(test_cases=test_cases, metrics=metrics)

    scores = {}
    for i, tr in enumerate(result.test_results):
        query_scores = {}
        for md in tr.metrics_data:
            query_scores[md.name] = md.score
        scores[TEST_QUERIES[i]] = query_scores
    return scores


def print_comparison(all_scores):
    """Print a side-by-side comparison table."""
    labels = list(all_scores.keys())

    print("\n" + "=" * 120)
    print("CROSS-SYSTEM COMPARISON — GoodMem vs Vectara")
    print("=" * 120)

    # Header
    header = f"{'Query':<35} {'Metric':<22}"
    for label in labels:
        header += f" {label:>18}"
    print(header)
    print("-" * 120)

    for query in TEST_QUERIES:
        short_q = query[:32] + "..." if len(query) > 35 else query
        for metric in ["Answer Relevancy", "Faithfulness", "Contextual Relevancy"]:
            q_col = short_q if metric == "Answer Relevancy" else ""
            row = f"{q_col:<35} {metric:<22}"
            for label in labels:
                s = all_scores[label].get(query, {}).get(metric, 0)
                row += f" {s:>18.2f}"
            print(row)
        print()

    # Print apples-to-apples summary
    print("=" * 120)
    print("APPLES-TO-APPLES SUMMARY (Contextual Relevancy only)")
    print("=" * 120)
    print(f"{'Query':<45} {'GoodMem':>10} {'Vectara':>10}  {'Setting':<15}")
    print("-" * 90)
    for query in TEST_QUERIES:
        short_q = query[:42] + "..." if len(query) > 45 else query
        gm3 = all_scores["GoodMem (top-3)"].get(query, {}).get("Contextual Relevancy", 0)
        v3 = all_scores["Vectara (top-3)"].get(query, {}).get("Contextual Relevancy", 0)
        print(f"{short_q:<45} {gm3:>10.2f} {v3:>10.2f}  top-3")
    print()
    for query in TEST_QUERIES:
        short_q = query[:42] + "..." if len(query) > 45 else query
        gm1 = all_scores["GoodMem (top-1)"].get(query, {}).get("Contextual Relevancy", 0)
        v1 = all_scores["Vectara (top-1)"].get(query, {}).get("Contextual Relevancy", 0)
        print(f"{short_q:<45} {gm1:>10.2f} {v1:>10.2f}  top-1")
    print()


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("GOODMEM vs VECTARA — Fair Cross-System RAG Comparison")
    print("=" * 70)
    print()
    print("Same docs, same queries, same generator — only retrieval differs.")
    print()
    print("Configurations:")
    print("  1. GoodMem (top-3)             — apples-to-apples with Vectara top-3")
    print("  2. Vectara (top-3)             — apples-to-apples with GoodMem top-3")
    print("  3. GoodMem (top-1)             — apples-to-apples with Vectara top-1")
    print("  4. Vectara (top-1)             — apples-to-apples with GoodMem top-1")
    print("  5. GoodMem + metadata filter   — targeted source filtering")

    # Set up both systems
    gm_client, space_id = setup_goodmem()
    vectara_queries_api, corpus_key = setup_vectara()

    # Build 5 pipelines
    print("\n🔧 Building pipelines...")

    # GoodMem top-3
    gm_retriever_broad = GoodMemRetriever(
        client=gm_client, spaces=[space_id], maximum_results=3,
    )
    goodmem_broad = GoodMemRAGPipeline(
        retriever=gm_retriever_broad, generator=openai_generator,
    )

    # Vectara top-3
    vectara_broad = VectaraRAGPipeline(
        queries_api=vectara_queries_api,
        corpus_key=corpus_key,
        generator=openai_generator,
        max_results=3,
    )

    # GoodMem top-1
    gm_retriever_precise = GoodMemRetriever(
        client=gm_client, spaces=[space_id], maximum_results=1,
    )
    goodmem_precise = GoodMemRAGPipeline(
        retriever=gm_retriever_precise, generator=openai_generator,
    )

    # Vectara top-1
    vectara_precise = VectaraRAGPipeline(
        queries_api=vectara_queries_api,
        corpus_key=corpus_key,
        generator=openai_generator,
        max_results=1,
    )

    # GoodMem + metadata filter
    goodmem_filtered = GoodMemFilteredPipeline(
        client=gm_client,
        space_id=space_id,
        generator=openai_generator,
        query_filters=TEST_QUERIES_WITH_FILTERS,
    )

    # Run evaluations
    print("\n🧪 Running 5 evaluations (this takes a few minutes)...\n")

    all_scores = {}
    all_scores["GoodMem (top-3)"] = run_eval(goodmem_broad, "GoodMem (top-3)")
    all_scores["Vectara (top-3)"] = run_eval(vectara_broad, "Vectara (top-3)")
    all_scores["GoodMem (top-1)"] = run_eval(goodmem_precise, "GoodMem (top-1)")
    all_scores["Vectara (top-1)"] = run_eval(vectara_precise, "Vectara (top-1)")
    all_scores["GoodMem+filter"] = run_eval(goodmem_filtered, "GoodMem + metadata filter")

    # Print comparison
    print_comparison(all_scores)

    print("💡 KEY TAKEAWAYS:")
    print("   • Top-3 comparisons are apples-to-apples: same top-k for both systems")
    print("   • Top-1 comparisons are apples-to-apples: same top-k for both systems")
    print("   • GoodMem + metadata filter shows what targeted retrieval can do")
    print("   • Use this framework to find the best retrieval config for your data")
    print()


if __name__ == "__main__":
    main()
