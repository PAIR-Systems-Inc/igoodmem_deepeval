import base64
import json
import os
import time
from typing import Any, Dict, List, Optional, Union

import requests

from deepeval.integrations.goodmem.types import (
    GoodMemEmbedder,
    GoodMemLLM,
    GoodMemMemory,
    GoodMemReranker,
    GoodMemRetrievalResult,
    GoodMemRetrievedChunk,
    GoodMemSpace,
)

# MIME type mapping for file uploads, matching the reference integration
_MIME_TYPES: Dict[str, str] = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "txt": "text/plain",
    "html": "text/html",
    "md": "text/markdown",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _get_mime_type(extension: str) -> Optional[str]:
    """Get MIME type from file extension."""
    ext = extension.lower().lstrip(".")
    return _MIME_TYPES.get(ext)


class GoodMemClient:
    """Client for interacting with the GoodMem API.

    GoodMem is a vector-based memory storage and semantic retrieval service.
    This client provides methods to create spaces, store memories (text or
    files), retrieve memories via semantic search, and manage stored content.

    Use this client to integrate GoodMem as a knowledge base for RAG
    evaluation workflows in deepeval. Retrieved memories can be used as
    ``retrieval_context`` in ``LLMTestCase`` for metrics like faithfulness,
    contextual relevancy, contextual precision, and contextual recall.

    Parameters
    ----------
    base_url : str
        The base URL of the GoodMem API server
        (e.g., ``"https://api.goodmem.ai"`` or ``"http://localhost:8080"``).
    api_key : str
        Your GoodMem API key for authentication (X-API-Key header).
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self, accept: str = "application/json") -> Dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": accept,
        }

    def _get(self, path: str, **kwargs: Any) -> requests.Response:
        resp = requests.get(
            f"{self.base_url}{path}",
            headers=self._headers(),
            **kwargs,
        )
        resp.raise_for_status()
        return resp

    def _post(
        self,
        path: str,
        body: Any,
        accept: str = "application/json",
        **kwargs: Any,
    ) -> requests.Response:
        resp = requests.post(
            f"{self.base_url}{path}",
            headers=self._headers(accept=accept),
            json=body,
            **kwargs,
        )
        resp.raise_for_status()
        return resp

    def _delete(self, path: str, **kwargs: Any) -> requests.Response:
        resp = requests.delete(
            f"{self.base_url}{path}",
            headers=self._headers(),
            **kwargs,
        )
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------
    # List Embedders (embedded/internal — used by create_space)
    # ------------------------------------------------------------------

    def list_embedders(self) -> List[GoodMemEmbedder]:
        """List available embedder models.

        Returns a list of embedders that can be used when creating spaces.
        This is used internally by :meth:`create_space` but is also exposed
        for discovery and configuration purposes.
        """
        resp = self._get("/v1/embedders")
        body = resp.json()
        embedders_raw = body if isinstance(body, list) else body.get("embedders", [])
        return [
            GoodMemEmbedder(
                embedder_id=e.get("embedderId") or e.get("id", ""),
                display_name=e.get("displayName") or e.get("name"),
                model_identifier=e.get("modelIdentifier") or e.get("model"),
            )
            for e in embedders_raw
        ]

    # ------------------------------------------------------------------
    # List Spaces (embedded/internal — used by create_space for reuse)
    # ------------------------------------------------------------------

    def list_spaces(self) -> List[Dict[str, Any]]:
        """List all spaces.

        Returns raw space dicts from the API. Used internally by
        :meth:`create_space` to detect and reuse existing spaces, but
        also exposed for discoverability.
        """
        resp = self._get("/v1/spaces")
        body = resp.json()
        return body if isinstance(body, list) else body.get("spaces", [])

    # ------------------------------------------------------------------
    # List Rerankers
    # ------------------------------------------------------------------

    def list_rerankers(self) -> List[GoodMemReranker]:
        """List available reranker models for post-processing retrieval results."""
        resp = self._get("/v1/rerankers")
        body = resp.json()
        rerankers_raw = body if isinstance(body, list) else body.get("rerankers", [])
        return [
            GoodMemReranker(
                reranker_id=r.get("rerankerId") or r.get("id", ""),
                display_name=r.get("displayName") or r.get("name"),
                model_identifier=r.get("modelIdentifier") or r.get("model"),
            )
            for r in rerankers_raw
        ]

    # ------------------------------------------------------------------
    # List LLMs
    # ------------------------------------------------------------------

    def list_llms(self) -> List[GoodMemLLM]:
        """List available LLM models for generating contextual responses."""
        resp = self._get("/v1/llms")
        body = resp.json()
        llms_raw = body if isinstance(body, list) else body.get("llms", [])
        return [
            GoodMemLLM(
                llm_id=l.get("llmId") or l.get("id", ""),
                display_name=l.get("displayName") or l.get("name"),
                model_identifier=l.get("model_identifier")
                or l.get("modelIdentifier")
                or l.get("model"),
            )
            for l in llms_raw
        ]

    # ------------------------------------------------------------------
    # Create Space
    # ------------------------------------------------------------------

    def create_space(
        self,
        name: str,
        embedder_id: str,
        chunk_size: int = 256,
        chunk_overlap: int = 25,
        keep_strategy: str = "KEEP_END",
        length_measurement: str = "CHARACTER_COUNT",
    ) -> GoodMemSpace:
        """Create a new space or reuse an existing one.

        A space is a logical container for organizing related memories,
        configured with an embedder that converts text to vector embeddings.

        If a space with the given ``name`` already exists, its ID is returned
        instead of creating a duplicate (matching the reference integration
        behavior).

        Parameters
        ----------
        name : str
            A unique name for the space.
        embedder_id : str
            The embedder model ID (from :meth:`list_embedders`).
        chunk_size : int
            Number of characters per chunk when splitting documents.
        chunk_overlap : int
            Number of overlapping characters between consecutive chunks.
        keep_strategy : str
            Where to attach the separator when splitting.
            One of ``"KEEP_END"``, ``"KEEP_START"``, ``"DISCARD"``.
        length_measurement : str
            How chunk size is measured.
            One of ``"CHARACTER_COUNT"``, ``"TOKEN_COUNT"``.
        """
        # Check for existing space with the same name
        try:
            spaces = self.list_spaces()
            for s in spaces:
                if s.get("name") == name:
                    return GoodMemSpace(
                        space_id=s.get("spaceId") or s.get("id", ""),
                        name=s.get("name", name),
                        embedder_id=embedder_id,
                        reused=True,
                    )
        except Exception:
            pass  # If listing fails, proceed to create

        request_body = {
            "name": name,
            "spaceEmbedders": [
                {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}
            ],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunk_size,
                    "chunkOverlap": chunk_overlap,
                    "separators": ["\n\n", "\n", ". ", " ", ""],
                    "keepStrategy": keep_strategy,
                    "separatorIsRegex": False,
                    "lengthMeasurement": length_measurement,
                }
            },
        }

        resp = self._post("/v1/spaces", request_body)
        body = resp.json()

        return GoodMemSpace(
            space_id=body.get("spaceId", ""),
            name=body.get("name", name),
            embedder_id=embedder_id,
            reused=False,
            chunking_config=request_body["defaultChunkingConfig"],
        )

    # ------------------------------------------------------------------
    # Create Memory
    # ------------------------------------------------------------------

    def create_memory(
        self,
        space_id: str,
        text_content: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GoodMemMemory:
        """Store a document as a new memory in a space.

        The memory is processed asynchronously --- chunked into searchable
        pieces and embedded into vectors. Provide either ``text_content``
        for plain text or ``file_path`` for a file (PDF, DOCX, images, etc.).
        If both are provided, the file takes priority.

        Parameters
        ----------
        space_id : str
            The space to store the memory in.
        text_content : str, optional
            Plain text content to store.
        file_path : str, optional
            Path to a file to upload as memory.
        metadata : dict, optional
            Key-value metadata to attach to the memory.

        Note
        ----
        Do NOT use ``author``, ``source``, or ``tags`` keys in metadata
        as per the integration requirements.
        """
        request_body: Dict[str, Any] = {"spaceId": space_id}

        if file_path is not None:
            # File upload — read and encode as base64
            ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
            mime_type = _get_mime_type(ext) or "application/octet-stream"

            with open(file_path, "rb") as f:
                file_bytes = f.read()

            if mime_type.startswith("text/"):
                request_body["contentType"] = mime_type
                request_body["originalContent"] = file_bytes.decode("utf-8")
            else:
                request_body["contentType"] = mime_type
                request_body["originalContentB64"] = base64.b64encode(
                    file_bytes
                ).decode("ascii")
        elif text_content is not None:
            request_body["contentType"] = "text/plain"
            request_body["originalContent"] = text_content
        else:
            raise ValueError(
                "No content provided. Please provide either text_content or file_path."
            )

        if metadata and len(metadata) > 0:
            request_body["metadata"] = metadata

        resp = self._post("/v1/memories", request_body)
        body = resp.json()

        return GoodMemMemory(
            memory_id=body.get("memoryId", ""),
            space_id=body.get("spaceId", space_id),
            processing_status=body.get("processingStatus", "PENDING"),
            content_type=request_body.get("contentType"),
            file_name=os.path.basename(file_path) if file_path else None,
        )

    # ------------------------------------------------------------------
    # Retrieve Memories
    # ------------------------------------------------------------------

    def retrieve_memories(
        self,
        query: str,
        space_ids: Union[str, List[str]],
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
        reranker_id: Optional[str] = None,
        llm_id: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
        llm_temperature: Optional[float] = None,
        chronological_resort: bool = False,
    ) -> GoodMemRetrievalResult:
        """Perform similarity-based semantic retrieval across spaces.

        Returns matching chunks ranked by relevance. Supports optional
        reranking and LLM-based post-processing.

        Parameters
        ----------
        query : str
            Natural language query for semantic similarity search.
        space_ids : str or list of str
            One or more space IDs to search across.
        max_results : int
            Maximum number of results to return.
        include_memory_definition : bool
            Whether to fetch full memory metadata alongside matched chunks.
        wait_for_indexing : bool
            If True, retry for up to 60 seconds when no results are found
            (useful when memories were just added and are still processing).
        reranker_id : str, optional
            Reranker model ID for improved result ordering.
        llm_id : str, optional
            LLM model ID for generating contextual responses.
        relevance_threshold : float, optional
            Minimum relevance score (0-1) for including results.
        llm_temperature : float, optional
            Creativity setting for LLM generation (0-2).
        chronological_resort : bool
            Reorder results by creation time instead of relevance score.
        """
        if isinstance(space_ids, str):
            space_ids = [space_ids]

        space_keys = [{"spaceId": sid} for sid in space_ids if sid]
        if not space_keys:
            raise ValueError("At least one space ID must be provided.")

        request_body: Dict[str, Any] = {
            "message": query,
            "spaceKeys": space_keys,
            "requestedSize": max_results,
            "fetchMemory": include_memory_definition,
        }

        # Post-processor config (reranker / LLM)
        if reranker_id or llm_id:
            config: Dict[str, Any] = {}
            if reranker_id:
                config["reranker_id"] = reranker_id
            if llm_id:
                config["llm_id"] = llm_id
            if relevance_threshold is not None:
                config["relevance_threshold"] = relevance_threshold
            if llm_temperature is not None:
                config["llm_temp"] = llm_temperature
            if max_results:
                config["max_results"] = max_results
            if chronological_resort:
                config["chronological_resort"] = True

            request_body["postProcessor"] = {
                "name": "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory",
                "config": config,
            }

        max_wait_ms = 60000
        poll_interval_s = 5.0
        start_time = time.time()

        while True:
            resp = self._post(
                "/v1/memories:retrieve",
                request_body,
                accept="application/x-ndjson",
            )

            result = self._parse_retrieval_response(resp, query)

            if result.total_results > 0 or not wait_for_indexing:
                return result

            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= max_wait_ms:
                return result

            time.sleep(poll_interval_s)

    def _parse_retrieval_response(
        self, resp: requests.Response, query: str
    ) -> GoodMemRetrievalResult:
        """Parse NDJSON / SSE response from the retrieve endpoint."""
        results: List[GoodMemRetrievedChunk] = []
        memories: List[Dict[str, Any]] = []
        result_set_id = ""
        abstract_reply = None

        response_text = resp.text
        for line in response_text.strip().split("\n"):
            json_str = line.strip()
            if not json_str:
                continue

            # Handle SSE format
            if json_str.startswith("data:"):
                json_str = json_str[5:].strip()
            if json_str.startswith("event:") or json_str == "":
                continue

            try:
                item = json.loads(json_str)

                if item.get("resultSetBoundary"):
                    result_set_id = item["resultSetBoundary"].get(
                        "resultSetId", ""
                    )
                elif item.get("memoryDefinition"):
                    memories.append(item["memoryDefinition"])
                elif item.get("abstractReply"):
                    abstract_reply = item["abstractReply"]
                elif item.get("retrievedItem"):
                    chunk_data = (
                        item["retrievedItem"].get("chunk", {}).get("chunk", {})
                    )
                    results.append(
                        GoodMemRetrievedChunk(
                            chunk_id=chunk_data.get("chunkId"),
                            chunk_text=chunk_data.get("chunkText"),
                            memory_id=chunk_data.get("memoryId"),
                            relevance_score=item["retrievedItem"]
                            .get("chunk", {})
                            .get("relevanceScore"),
                            memory_index=item["retrievedItem"]
                            .get("chunk", {})
                            .get("memoryIndex"),
                        )
                    )
            except (json.JSONDecodeError, KeyError):
                continue

        return GoodMemRetrievalResult(
            result_set_id=result_set_id,
            results=results,
            memories=memories,
            total_results=len(results),
            query=query,
            abstract_reply=abstract_reply,
        )

    # ------------------------------------------------------------------
    # Get Memory
    # ------------------------------------------------------------------

    def get_memory(
        self,
        memory_id: str,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """Fetch a specific memory record by its ID.

        Parameters
        ----------
        memory_id : str
            The UUID of the memory to fetch.
        include_content : bool
            Whether to fetch the original document content in addition
            to metadata.
        """
        resp = self._get(f"/v1/memories/{memory_id}")
        result: Dict[str, Any] = {"memory": resp.json()}

        if include_content:
            try:
                content_resp = self._get(f"/v1/memories/{memory_id}/content")
                result["content"] = content_resp.json()
            except Exception as e:
                result["content_error"] = f"Failed to fetch content: {e}"

        return result

    # ------------------------------------------------------------------
    # Delete Memory
    # ------------------------------------------------------------------

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Permanently delete a memory and its associated chunks and vectors.

        Parameters
        ----------
        memory_id : str
            The UUID of the memory to delete.
        """
        self._delete(f"/v1/memories/{memory_id}")
        return {
            "success": True,
            "memory_id": memory_id,
            "message": "Memory deleted successfully",
        }

    # ------------------------------------------------------------------
    # Convenience: retrieve as retrieval_context for LLMTestCase
    # ------------------------------------------------------------------

    def retrieve_as_context(
        self,
        query: str,
        space_ids: Union[str, List[str]],
        max_results: int = 5,
        wait_for_indexing: bool = True,
        **kwargs: Any,
    ) -> List[str]:
        """Retrieve memories and return chunk texts as a list of strings.

        This is a convenience method designed for use with deepeval's
        ``LLMTestCase.retrieval_context`` field. It calls
        :meth:`retrieve_memories` and extracts the chunk text from each
        result.

        Parameters
        ----------
        query : str
            Natural language query for semantic similarity search.
        space_ids : str or list of str
            Space IDs to search across.
        max_results : int
            Maximum number of results.
        wait_for_indexing : bool
            Whether to wait for indexing to complete.
        **kwargs
            Additional keyword arguments passed to :meth:`retrieve_memories`.

        Returns
        -------
        list of str
            Chunk texts from matched memories, suitable for use as
            ``retrieval_context`` in ``LLMTestCase``.
        """
        result = self.retrieve_memories(
            query=query,
            space_ids=space_ids,
            max_results=max_results,
            wait_for_indexing=wait_for_indexing,
            **kwargs,
        )
        return [
            chunk.chunk_text
            for chunk in result.results
            if chunk.chunk_text is not None
        ]
