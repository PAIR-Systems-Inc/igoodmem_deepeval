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

    def __init__(self, config: GoodMemEvalConfig):
        configuration = Configuration()
        configuration.host = config.base_url.rstrip("/")
        configuration.api_key = {"ApiKeyAuth": config.api_key}

        # NOTE: goodmem-client exposes timeout via ApiClient; we keep it simple here.
        self._api_client = ApiClient(configuration=configuration)
        self._spaces_api = SpacesApi(self._api_client)
        self._memories_api = MemoriesApi(self._api_client)
        self._stream_client = MemoryStreamClient(self._api_client)

    # Spaces -----------------------------------------------------------------
    def create_space(
        self,
        space_name: str,
        embedder_id: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        public_read: Optional[bool] = None,
        default_chunking_config: Optional[Dict[str, Any]] = None,
    ) -> Space:
        """
        Create a new GoodMem space with a simple configuration surface suitable
        for evaluation and benchmarks.
        """
        if default_chunking_config is None:
            # Provide a sane default recursive chunking configuration, mirroring
            # the examples in the official GoodMem docs.
            default_chunking_config = {
                "recursive": {
                    "chunkSize": 1000,
                    "chunkOverlap": 200,
                    "separators": ["\n\n", "\n", ".", " "],
                    "keepStrategy": "KEEP_END",
                    "separatorIsRegex": False,
                    "lengthMeasurement": "CHARACTER_COUNT",
                }
            }

        request = SpaceCreationRequest(
            name=space_name,
            labels=labels or {},
            public_read=public_read if public_read is not None else False,
            default_chunking_config=default_chunking_config,
        )
        # Attach embedder if provided; we keep this minimal to avoid leaking SDK details.
        if embedder_id is not None:
            request.space_embedders = [
                {
                    "embedderId": embedder_id,
                    "defaultRetrievalWeight": 1.0,
                }
            ]

        return self._spaces_api.create_space(space_creation_request=request)

    def list_spaces(self) -> List[Space]:
        response = self._spaces_api.list_spaces()
        # goodmem-client returns a ListSpacesResponse; we expose just the spaces list.
        spaces: Sequence[Space] = getattr(response, "spaces", []) or []
        return list(spaces)

    # Memories ---------------------------------------------------------------
    def create_memory(
        self,
        space_id: str,
        text_content: str,
        content_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """
        Create a new memory within a space. This uses the JSON form of the API
        and is sufficient for most evaluation/benchmark scenarios.
        """
        request = MemoryCreationRequest(
            space_id=space_id,
            original_content=text_content,
            content_type=content_type,
            metadata=metadata or {},
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
        reranker_id: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute an advanced semantic retrieval against one or more spaces.

        This uses the POST /v1/memories:retrieve endpoint via the
        `retrieve_memory_advanced` method, and returns a simple dict structure
        that is easy to convert into DeepEval retrieval_context lists.
        """
        # NOTE: current implementation uses MemoryStreamClient which does not
        # accept explicit reranker or relevance-threshold parameters. Those
        # are kept in the signature for future extension but are currently
        # not forwarded.

        space_keys: List[SpaceKey] = []
        for space_id in spaces:
            space_key = SpaceKey(space_id=space_id)
            if metadata_filters:
                # Caller can pass a pre-built filter expression instead of a dict
                # by setting a special key. Otherwise, we keep this simple and
                # assume metadata_filters contains a raw filter string.
                filter_expr = metadata_filters.get("filter")
                if filter_expr:
                    space_key.filter = filter_expr
            space_keys.append(space_key)

        # Use the streaming client for advanced retrieval; this is the
        # officially supported path in goodmem-client.
        events = list(
            self._stream_client.retrieve_memory_stream(
                message=query,
                space_keys=space_keys,
                requested_size=maximum_results,
                fetch_memory=include_memory_definition,
                fetch_memory_content=False,
            )
        )

        results: List[Dict[str, Any]] = []
        for event in events:
            retrieved_item = getattr(event, "retrieved_item", None)
            if not retrieved_item or not retrieved_item.chunk:
                continue
            chunk = retrieved_item.chunk
            memory_def = getattr(event, "memory_definition", None)

             # chunk.chunk is an SDK model field, not a plain dict; handle both
             # object-with-attribute and mapping styles defensively.
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

