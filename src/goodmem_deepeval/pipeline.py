from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from .retriever import GoodMemRetriever, GoodMemChunk

RAGPipeline = Callable[[str], Tuple[str, List[str]]]


class GoodMemRAGPipeline:
    """
    Simple RAG pipeline abstraction on top of GoodMemRetriever and a
    user-provided answer generator.

    The generator is any callable that accepts (query, retrieval_context)
    and returns an answer string.
    """

    def __init__(
        self,
        retriever: GoodMemRetriever,
        generator: Callable[[str, Sequence[str]], str],
    ) -> None:
        self._retriever = retriever
        self._generator = generator

    def __call__(self, query: str) -> Tuple[str, List[str]]:
        chunks: Sequence[GoodMemChunk] = self._retriever.retrieve(query)
        retrieval_context: List[str] = self._retriever.to_text_list(chunks)
        answer = self._generator(query, retrieval_context)
        return answer, retrieval_context

