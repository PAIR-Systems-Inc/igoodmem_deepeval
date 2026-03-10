from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .client import GoodMemEvalClient


@dataclass
class GoodMemChunk:
    content: str
    memory_id: Optional[str]
    space_id: Optional[str]
    score: Optional[float]
    metadata: Dict[str, Any]

    @property
    def source(self) -> Optional[str]:
        return self.metadata.get("source")

    @property
    def author(self) -> Optional[str]:
        return self.metadata.get("author")

    @property
    def tags(self) -> Sequence[str]:
        value = self.metadata.get("tags") or []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # allow comma-separated tags for convenience
            return [t.strip() for t in value.split(",") if t.strip()]
        return []


class GoodMemRetriever:
    """
    Thin wrapper over GoodMemEvalClient.retrieve_memories that returns
    structured GoodMemChunk objects and DeepEval-ready retrieval_context lists.
    """

    def __init__(
        self,
        client: GoodMemEvalClient,
        spaces: Iterable[str],
        maximum_results: int = 5,
        relevance_threshold: Optional[float] = None,
        reranker_id: Optional[str] = None,
        metadata_filter: Optional[str] = None,
    ) -> None:
        self._client = client
        self._spaces = list(spaces)
        self._maximum_results = maximum_results
        self._relevance_threshold = relevance_threshold
        self._reranker_id = reranker_id
        self._metadata_filter = metadata_filter

    def retrieve(self, query: str) -> List[GoodMemChunk]:
        retrieval = self._client.retrieve_memories(
            query=query,
            spaces=self._spaces,
            maximum_results=self._maximum_results,
            reranker_id=self._reranker_id,
            relevance_threshold=self._relevance_threshold,
            metadata_filters={"filter": self._metadata_filter} if self._metadata_filter else None,
        )
        results: Sequence[Dict[str, Any]] = retrieval.get("results", []) or []
        chunks: List[GoodMemChunk] = []
        for item in results:
            chunks.append(
                GoodMemChunk(
                    content=item.get("content", ""),
                    memory_id=item.get("memory_id"),
                    space_id=item.get("space_id"),
                    score=item.get("score"),
                    metadata=item.get("metadata") or {},
                )
            )
        return chunks

    @staticmethod
    def to_text_list(chunks: Sequence[GoodMemChunk]) -> List[str]:
        """
        Convert a list of GoodMemChunk into a retrieval_context list[str]
        suitable for DeepEval RAG metrics.
        """
        return [chunk.content for chunk in chunks]

