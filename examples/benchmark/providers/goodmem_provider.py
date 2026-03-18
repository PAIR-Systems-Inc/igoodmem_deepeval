"""GoodMem retrieval provider for the benchmark framework."""
from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence

from goodmem_client.api.embedders_api import EmbeddersApi
from goodmem_client.api_client import ApiClient as GoodMemApiClient
from goodmem_client.configuration import Configuration as GoodMemConfiguration

from goodmem_deepeval import (
    GoodMemEvalClient,
    GoodMemRAGPipeline,
    GoodMemRetriever,
)

from .base import Generator, RAGPipeline, RetrievalProvider
from ..datasets import BenchmarkDocument

SPACE_NAME = "scaled-benchmark"
BATCH_SIZE = 50
BATCH_DELAY = 1.0  # seconds between batches
INDEX_WAIT = 10  # seconds to wait after indexing


class GoodMemProvider(RetrievalProvider):
    """Benchmarks GoodMem retrieval using the goodmem_deepeval integration."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self._base_url = base_url or os.environ.get(
            "GOODMEM_BASE_URL", "http://localhost:8080"
        )
        self._api_key = api_key or os.environ.get("GOODMEM_API_KEY")
        if not self._api_key:
            raise RuntimeError("GOODMEM_API_KEY is required.")

        self._client = GoodMemEvalClient(
            base_url=self._base_url, api_key=self._api_key
        )
        self._space_id: Optional[str] = None

    @property
    def name(self) -> str:
        return "GoodMem"

    def _get_embedder_id(self) -> str:
        """Get the first available embedder from the GoodMem instance."""
        cfg = GoodMemConfiguration()
        cfg.host = self._base_url
        cfg.api_key = {"ApiKeyAuth": self._api_key}
        api_client = GoodMemApiClient(configuration=cfg)
        embedders = EmbeddersApi(api_client).list_embedders().embedders
        if not embedders:
            raise RuntimeError("No embedders configured in GoodMem.")
        return embedders[0].embedder_id

    def setup(self, documents: List[BenchmarkDocument]) -> None:
        """Create a space and index all documents in batches."""
        # Use a versioned space name for idempotency
        space_name = f"{SPACE_NAME}-{len(documents)}"

        # Check for existing space with matching name
        spaces = self._client.list_spaces()
        existing = next((s for s in spaces if s.name == space_name), None)

        if existing:
            self._space_id = existing.space_id
            print(f"  📂 Reusing existing space: {self._space_id}")
            return

        # Create new space
        embedder_id = self._get_embedder_id()
        space = self._client.create_space(
            space_name=space_name,
            embedder=embedder_id,
            chunk_size=512,
            chunk_overlap=50,
        )
        self._space_id = space.space_id
        print(f"  📂 Created space: {self._space_id}")

        # Batch-index documents
        total = len(documents)
        for i in range(0, total, BATCH_SIZE):
            batch = documents[i : i + BATCH_SIZE]
            for doc in batch:
                self._client.create_memory(
                    space=self._space_id,
                    text_content=doc.text,
                    source=doc.metadata.get("source", ""),
                    author=doc.metadata.get("title", ""),
                    additional_metadata=doc.metadata,
                )
            indexed = min(i + BATCH_SIZE, total)
            print(f"  📝 Indexed {indexed}/{total} documents")
            if indexed < total:
                time.sleep(BATCH_DELAY)

        print(f"  ⏳ Waiting {INDEX_WAIT}s for indexing to complete...")
        time.sleep(INDEX_WAIT)
        print("  ✅ GoodMem ready")

    def make_pipeline(
        self,
        generator: Generator,
        top_k: int,
        metadata_filter: Optional[str] = None,
    ) -> RAGPipeline:
        if not self._space_id:
            raise RuntimeError("Call setup() before make_pipeline().")

        retriever = GoodMemRetriever(
            client=self._client,
            spaces=[self._space_id],
            maximum_results=top_k,
            metadata_filter=metadata_filter,
        )
        return GoodMemRAGPipeline(retriever=retriever, generator=generator)

    def teardown(self) -> None:
        """No-op for now — spaces are reused across runs."""
        pass
