"""Abstract base class for retrieval providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Tuple

from ..datasets import BenchmarkDocument

# RAGPipeline type: (query) -> (answer, retrieval_context)
RAGPipeline = Callable[[str], Tuple[str, List[str]]]
Generator = Callable[[str, Sequence[str]], str]


class RetrievalProvider(ABC):
    """
    Abstract base for any retrieval system that can be benchmarked.

    To add a new provider (e.g. Pinecone, Weaviate, Qdrant):
      1. Subclass RetrievalProvider
      2. Implement setup(), make_pipeline(), and teardown()
      3. Register it in providers/__init__.py
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this provider (e.g. 'GoodMem', 'Vectara')."""

    @abstractmethod
    def setup(self, documents: List[BenchmarkDocument]) -> None:
        """
        Index all documents into this provider's storage.

        Should be idempotent: if the corpus/space already exists with the
        correct document count, skip re-indexing.
        """

    @abstractmethod
    def make_pipeline(
        self,
        generator: Generator,
        top_k: int,
        metadata_filter: Optional[str] = None,
    ) -> RAGPipeline:
        """
        Return a RAGPipeline callable configured with the given parameters.

        The pipeline must accept (query: str) and return (answer: str,
        retrieval_context: list[str]).
        """

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources (delete corpus/space). Called at end of benchmark."""
