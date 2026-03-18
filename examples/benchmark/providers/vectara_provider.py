"""Vectara retrieval provider for the benchmark framework."""
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

from .base import Generator, RAGPipeline, RetrievalProvider
from ..datasets import BenchmarkDocument

CORPUS_KEY = "scaled-benchmark"
BATCH_SIZE = 50
BATCH_DELAY = 1.0
INDEX_WAIT = 10
MAX_PART_SIZE = 15000  # Vectara max is 16,384 chars per document part; leave margin


class _VectaraPipeline:
    """Wraps Vectara query API into a RAGPipeline callable."""

    def __init__(self, queries_api, corpus_key: str, generator: Generator, top_k: int):
        self._queries_api = queries_api
        self._corpus_key = corpus_key
        self._generator = generator
        self._top_k = top_k

    def __call__(self, query: str) -> Tuple[str, List[str]]:
        from vectara_client import QueryCorpusRequest, SearchCorpusParameters

        request = QueryCorpusRequest(
            query=query,
            search=SearchCorpusParameters(limit=self._top_k),
        )
        response = self._queries_api.query_corpus(
            corpus_key=self._corpus_key,
            query_corpus_request=request,
        )

        retrieval_context = []
        if response.search_results:
            for result in response.search_results[: self._top_k]:
                if result.text:
                    retrieval_context.append(result.text)

        answer = self._generator(query, retrieval_context)
        return answer, retrieval_context


class VectaraProvider(RetrievalProvider):
    """Benchmarks Vectara retrieval."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("VECTARA_API_KEY")
        if not self._api_key:
            raise RuntimeError("VECTARA_API_KEY is required.")
        self._queries_api = None
        self._corpus_key = CORPUS_KEY

    @property
    def name(self) -> str:
        return "Vectara"

    def setup(self, documents: List[BenchmarkDocument]) -> None:
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

        vcfg = VectaraConfiguration()
        vcfg.api_key["ApiKeyAuth"] = self._api_key
        api_client = VectaraApiClient(configuration=vcfg)

        corpora_api = CorporaApi(api_client)
        index_api = IndexApi(api_client)
        self._queries_api = QueriesApi(api_client)

        # Use a versioned corpus key for idempotency
        self._corpus_key = f"{CORPUS_KEY}-{len(documents)}"

        try:
            corpus = corpora_api.get_corpus(corpus_key=self._corpus_key)
            # Verify corpus actually has documents by doing a test query
            test_resp = self._queries_api.query_corpus(
                corpus_key=self._corpus_key,
                query_corpus_request=QueryCorpusRequest(
                    query="test",
                    search=SearchCorpusParameters(limit=1),
                ),
            )
            if test_resp.search_results and len(test_resp.search_results) > 0:
                print(f"  📂 Reusing existing corpus: {self._corpus_key}")
                return
            else:
                print(f"  ⚠️  Corpus exists but is empty. Deleting and recreating...")
                corpora_api.delete_corpus(corpus_key=self._corpus_key)
        except Exception:
            pass

        corpus = corpora_api.create_corpus(
            create_corpus_request=CreateCorpusRequest(
                key=self._corpus_key,
                name=f"Scaled Benchmark ({len(documents)} docs)",
                description="Auto-created by scaled RAG benchmark",
            )
        )
        print(f"  📂 Created corpus: {self._corpus_key}")

        # Vectara corpus creation is async — poll until ready
        print("  ⏳ Waiting for corpus to be available...")
        for attempt in range(10):
            try:
                corpora_api.get_corpus(corpus_key=self._corpus_key)
                print("  ✅ Corpus is ready")
                break
            except Exception:
                time.sleep(3)
        else:
            raise RuntimeError(f"Corpus {self._corpus_key} not ready after 30s")

        total = len(documents)
        for i in range(0, total, BATCH_SIZE):
            batch = documents[i : i + BATCH_SIZE]
            for doc in batch:
                # Split large documents into multiple parts (Vectara limit: 16KB per part)
                text = doc.text
                parts = []
                while text:
                    chunk = text[:MAX_PART_SIZE]
                    text = text[MAX_PART_SIZE:]
                    parts.append(CoreDocumentPart(text=chunk))

                core_doc = CoreDocument(
                    id=doc.doc_id,
                    type="core",
                    metadata=doc.metadata,
                    document_parts=parts,
                )
                # Retry indexing in case corpus is still propagating
                for retry in range(3):
                    try:
                        index_api.create_corpus_document(
                            corpus_key=self._corpus_key,
                            create_document_request=CreateDocumentRequest(
                                actual_instance=core_doc,
                            ),
                        )
                        break
                    except Exception as e:
                        if retry < 2 and "404" in str(e):
                            time.sleep(5)
                        else:
                            raise
            indexed = min(i + BATCH_SIZE, total)
            print(f"  📝 Indexed {indexed}/{total} documents")
            if indexed < total:
                time.sleep(BATCH_DELAY)

        print(f"  ⏳ Waiting {INDEX_WAIT}s for indexing to complete...")
        time.sleep(INDEX_WAIT)
        print("  ✅ Vectara ready")

    def make_pipeline(
        self,
        generator: Generator,
        top_k: int,
        metadata_filter: Optional[str] = None,
    ) -> RAGPipeline:
        if not self._queries_api:
            raise RuntimeError("Call setup() before make_pipeline().")
        return _VectaraPipeline(
            queries_api=self._queries_api,
            corpus_key=self._corpus_key,
            generator=generator,
            top_k=top_k,
        )

    def teardown(self) -> None:
        pass
