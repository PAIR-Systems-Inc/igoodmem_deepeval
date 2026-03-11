from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from goodmem_client import ApiClient, Configuration
from goodmem_client.api.memories_api import MemoriesApi
from goodmem_client.api.spaces_api import SpacesApi
from goodmem_client.models import (
    Memory,
    MemoryCreationRequest,
    Space,
    SpaceCreationRequest,
    SpaceKey,
)
from goodmem_client.streaming import MemoryStreamClient


@dataclass
class GoodMemEvalConfig:
    base_url: str
    api_key: str
    timeout: Optional[float] = None


class GoodMemEvalClient:
    """
    Evaluation-focused wrapper around the GoodMem Python SDK.

    This client intentionally exposes a narrow set of operations that are
    commonly needed when building DeepEval test cases:

    - Create/list spaces
    - Create/list/get/delete memories
    - Retrieve memories with semantic search and optional filters/rerankers
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[GoodMemEvalConfig] = None,
    ):
        if config is not None:
            base_url = config.base_url
            api_key = config.api_key
        if not base_url or not api_key:
            raise ValueError("base_url and api_key are required (pass directly or via config).")

        configuration = Configuration()
        configuration.host = base_url.rstrip("/")
        configuration.api_key = {"ApiKeyAuth": api_key}

        self._api_client = ApiClient(configuration=configuration)
        self._spaces_api = SpacesApi(self._api_client)
        self._memories_api = MemoriesApi(self._api_client)
        self._stream_client = MemoryStreamClient(self._api_client)

    # Spaces -----------------------------------------------------------------
    def create_space(
        self,
        space_name: str,
        embedder: Optional[str] = None,
        chunk_size: int = 256,
        chunk_overlap: int = 25,
        keep_separator_strategy: str = "KEEP_END",
        length_measurement: str = "CHARACTER_COUNT",
        labels: Optional[Dict[str, str]] = None,
        public_read: Optional[bool] = None,
    ) -> Space:
        """
        Create a new GoodMem space with explicit chunking parameters suitable
        for evaluation and benchmarks.
        """
        default_chunking_config = {
            "recursive": {
                "chunkSize": chunk_size,
                "chunkOverlap": chunk_overlap,
                "separators": ["\n\n", "\n", ".", " "],
                "keepStrategy": keep_separator_strategy,
                "separatorIsRegex": False,
                "lengthMeasurement": length_measurement,
            }
        }

        request = SpaceCreationRequest(
            name=space_name,
            labels=labels or {},
            public_read=public_read if public_read is not None else False,
            default_chunking_config=default_chunking_config,
        )
        if embedder is not None:
            request.space_embedders = [
                {
                    "embedderId": embedder,
                    "defaultRetrievalWeight": 1.0,
                }
            ]

        return self._spaces_api.create_space(space_creation_request=request)

    def list_spaces(self) -> List[Space]:
        response = self._spaces_api.list_spaces()
        spaces: Sequence[Space] = getattr(response, "spaces", []) or []
        return list(spaces)

    # Memories ---------------------------------------------------------------
    def create_memory(
        self,
        space: str,
        text_content: str,
        source: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        content_type: str = "text/plain",
    ) -> Memory:
        """
        Create a new memory within a space. Metadata fields (source, author,
        tags) are promoted to top-level params for convenience.
        """
        metadata: Dict[str, Any] = dict(additional_metadata or {})
        if source is not None:
            metadata["source"] = source
        if author is not None:
            metadata["author"] = author
        if tags is not None:
            metadata["tags"] = tags

        request = MemoryCreationRequest(
            space_id=space,
            original_content=text_content,
            content_type=content_type,
            metadata=metadata,
        )
        return self._memories_api.create_memory(memory_creation_request=request)

    def list_memories(self, space_id: str, page_size: int = 100) -> List[Memory]:
        response = self._memories_api.list_memories(space_id=space_id, page_size=page_size)
        memories: Sequence[Memory] = getattr(response, "memories", []) or []
        return list(memories)

    def get_memory(self, memory_id: str, include_content: bool = True) -> Memory:
        return self._memories_api.get_memory(
            id=memory_id,
            include_content=include_content,
            include_processing_history=False,
        )

    def delete_memory(self, memory_id: str) -> None:
        self._memories_api.delete_memory(id=memory_id)

    # Retrieval --------------------------------------------------------------
    def retrieve_memories(
        self,
        query: str,
        spaces: Iterable[str],
        maximum_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
        reranker: Optional[str] = None,
        llm: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
        llm_temperature: Optional[float] = None,
        chronological_resort: bool = False,
        metadata_filters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a semantic retrieval against one or more spaces using the
        ChatPostProcessor streaming endpoint.
        """
        space_keys: List[SpaceKey] = []
        for space_id in spaces:
            space_key = SpaceKey(space_id=space_id)
            if metadata_filters:
                filter_expr = metadata_filters.get("filter")
                if filter_expr:
                    space_key.filter = filter_expr
            space_keys.append(space_key)

        events = list(
            self._stream_client.retrieve_memory_stream_chat(
                message=query,
                space_keys=space_keys,
                requested_size=maximum_results,
                fetch_memory=include_memory_definition,
                fetch_memory_content=False,
                pp_reranker_id=reranker,
                pp_llm_id=llm,
                pp_relevance_threshold=relevance_threshold,
                pp_llm_temp=llm_temperature,
                pp_chronological_resort=chronological_resort if chronological_resort else None,
            )
        )

        results: List[Dict[str, Any]] = []
        for event in events:
            retrieved_item = getattr(event, "retrieved_item", None)
            if not retrieved_item or not retrieved_item.chunk:
                continue
            chunk = retrieved_item.chunk
            memory_def = getattr(event, "memory_definition", None)

            raw_chunk = getattr(chunk, "chunk", None)
            if isinstance(raw_chunk, dict):
                chunk_text = raw_chunk.get("chunkText", "")
            else:
                chunk_text = getattr(raw_chunk, "chunkText", "") if raw_chunk is not None else ""

            metadata = getattr(memory_def, "metadata", {}) or {}

            results.append(
                {
                    "memory_id": getattr(chunk, "memory_id", None),
                    "space_id": getattr(chunk, "space_id", None),
                    "score": getattr(chunk, "relevance_score", None),
                    "content": chunk_text,
                    "metadata": metadata,
                }
            )

        return {
            "query": query,
            "results": results,
        }
