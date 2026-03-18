"""Weaviate retrieval provider for the benchmark framework."""
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

from .base import Generator, RAGPipeline, RetrievalProvider
from ..datasets import BenchmarkDocument

COLLECTION_PREFIX = "ScaledBenchmark"
BATCH_SIZE = 100
CHUNK_SIZE = 512   # Match GoodMem's chunk size for apples-to-apples comparison
CHUNK_OVERLAP = 50  # Match GoodMem's chunk overlap


class _WeaviatePipeline:
    """Wraps Weaviate query API into a RAGPipeline callable."""

    def __init__(self, client, collection_name: str, generator: Generator, top_k: int):
        self._client = client
        self._collection_name = collection_name
        self._generator = generator
        self._top_k = top_k

    def __call__(self, query: str) -> Tuple[str, List[str]]:
        collection = self._client.collections.get(self._collection_name)
        response = collection.query.near_text(
            query=query,
            limit=self._top_k,
            return_properties=["text", "doc_id"],
        )

        retrieval_context = []
        for obj in response.objects:
            text = obj.properties.get("text", "")
            if text:
                retrieval_context.append(text)

        answer = self._generator(query, retrieval_context)
        return answer, retrieval_context


class WeaviateProvider(RetrievalProvider):
    """Benchmarks Weaviate retrieval."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self._url = url or os.environ.get("WEAVIATE_URL", "")
        self._api_key = api_key or os.environ.get("WEAVIATE_API_KEY", "")
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._url:
            raise RuntimeError("WEAVIATE_URL is required.")
        if not self._api_key:
            raise RuntimeError("WEAVIATE_API_KEY is required.")
        if not self._openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required (for Weaviate text2vec-openai vectorizer).")
        self._client = None
        self._collection_name = COLLECTION_PREFIX

    @property
    def name(self) -> str:
        return "Weaviate"

    def _connect(self):
        """Establish connection to Weaviate Cloud."""
        import weaviate
        from weaviate.classes.init import Auth

        self._client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self._url,
            auth_credentials=Auth.api_key(self._api_key),
            headers={"X-OpenAI-Api-Key": self._openai_api_key},
        )

    def setup(self, documents: List[BenchmarkDocument]) -> None:
        import weaviate.classes.config as wvc

        self._connect()

        # Use a versioned collection name for idempotency
        self._collection_name = f"{COLLECTION_PREFIX}_{len(documents)}"

        # Check if collection exists and has data
        if self._client.collections.exists(self._collection_name):
            collection = self._client.collections.get(self._collection_name)
            count = collection.aggregate.over_all(total_count=True).total_count
            if count > 0:
                print(f"  📂 Reusing existing collection: {self._collection_name} ({count} chunks)")
                return
            else:
                print(f"  ⚠️  Collection exists but is empty. Deleting and recreating...")
                self._client.collections.delete(self._collection_name)

        # Create collection with text2vec-openai vectorizer
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._client.collections.create(
                name=self._collection_name,
                vectorizer_config=wvc.Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small",
                ),
                properties=[
                    wvc.Property(name="text", data_type=wvc.DataType.TEXT),
                    wvc.Property(
                        name="doc_id",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                ],
            )
        print(f"  📂 Created collection: {self._collection_name}")

        # Pre-chunk and batch index
        collection = self._client.collections.get(self._collection_name)
        total_chunks = 0
        failed = 0

        for doc_idx, doc in enumerate(documents):
            # Pre-chunk to match GoodMem/Vectara (512 chars, 50 overlap)
            text = doc.text
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))
                chunks.append(text[start:end])
                start += CHUNK_SIZE - CHUNK_OVERLAP

            # Batch insert chunks
            with collection.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
                for chunk in chunks:
                    batch.add_object(properties={
                        "text": chunk,
                        "doc_id": doc.doc_id,
                    })

            # Check for errors
            if collection.batch.failed_objects:
                failed += len(collection.batch.failed_objects)

            total_chunks += len(chunks)
            if (doc_idx + 1) % 10 == 0 or doc_idx == len(documents) - 1:
                print(f"  📝 Indexed {doc_idx + 1}/{len(documents)} documents ({total_chunks} chunks)")

        if failed:
            print(f"  ⚠️  {failed} chunks failed to index")

        print(f"  ✅ Weaviate ready ({total_chunks} chunks indexed)")

    def make_pipeline(
        self,
        generator: Generator,
        top_k: int,
        metadata_filter: Optional[str] = None,
    ) -> RAGPipeline:
        if not self._client:
            raise RuntimeError("Call setup() before make_pipeline().")
        return _WeaviatePipeline(
            client=self._client,
            collection_name=self._collection_name,
            generator=generator,
            top_k=top_k,
        )

    def teardown(self) -> None:
        if self._client:
            self._client.close()
