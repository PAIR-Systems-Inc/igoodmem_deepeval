from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from goodmem_deepeval.client import GoodMemEvalClient, GoodMemEvalConfig


class TestClientConstructor:
    def test_direct_kwargs(self):
        with patch("goodmem_deepeval.client.ApiClient"):
            client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="key")
            assert client is not None

    def test_via_config(self):
        with patch("goodmem_deepeval.client.ApiClient"):
            config = GoodMemEvalConfig(base_url="http://localhost:8080", api_key="key")
            client = GoodMemEvalClient(config=config)
            assert client is not None

    def test_missing_params_raises(self):
        with pytest.raises(ValueError, match="base_url and api_key are required"):
            GoodMemEvalClient()

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError):
            GoodMemEvalClient(base_url="http://localhost:8080")


class TestCreateSpace:
    def test_creates_space_with_explicit_params(self):
        with patch("goodmem_deepeval.client.ApiClient"):
            client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="key")
            mock_space = MagicMock(space_id="sp-1")
            client._spaces_api.create_space = MagicMock(return_value=mock_space)

            result = client.create_space(
                space_name="bench",
                embedder="bge-small",
                chunk_size=512,
                chunk_overlap=50,
                keep_separator_strategy="KEEP_END",
                length_measurement="CHARACTER_COUNT",
            )

            assert result.space_id == "sp-1"
            call_kwargs = client._spaces_api.create_space.call_args
            request = call_kwargs.kwargs["space_creation_request"]
            chunking = request.default_chunking_config.recursive
            assert chunking.chunk_size == 512
            assert chunking.chunk_overlap == 50
            assert chunking.keep_strategy.value == "KEEP_END"
            assert chunking.length_measurement.value == "CHARACTER_COUNT"

    def test_embedder_attached(self):
        with patch("goodmem_deepeval.client.ApiClient"):
            client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="key")
            client._spaces_api.create_space = MagicMock(return_value=MagicMock())

            client.create_space(space_name="test", embedder="emb-1")

            request = client._spaces_api.create_space.call_args.kwargs["space_creation_request"]
            assert request.space_embedders[0].embedder_id == "emb-1"


class TestCreateMemory:
    def test_metadata_built_from_top_level_params(self):
        with patch("goodmem_deepeval.client.ApiClient"):
            client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="key")
            client._memories_api.create_memory = MagicMock(return_value=MagicMock())

            client.create_memory(
                space="sp-1",
                text_content="hello",
                source="docs",
                author="me",
                tags="a,b",
                additional_metadata={"custom": "val"},
            )

            request = client._memories_api.create_memory.call_args.kwargs["memory_creation_request"]
            assert request.space_id == "sp-1"
            assert request.original_content == "hello"
            assert request.metadata["source"] == "docs"
            assert request.metadata["author"] == "me"
            assert request.metadata["tags"] == "a,b"
            assert request.metadata["custom"] == "val"


class TestRetrieveMemories:
    def test_forwards_reranker_and_llm_params(self):
        with patch("goodmem_deepeval.client.ApiClient"):
            client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="key")
            client._stream_client.retrieve_memory_stream_chat = MagicMock(return_value=iter([]))

            client.retrieve_memories(
                query="test",
                spaces=["sp-1"],
                reranker="reranker-1",
                llm="llm-1",
                relevance_threshold=0.5,
                llm_temperature=0.7,
                chronological_resort=True,
            )

            call_kwargs = client._stream_client.retrieve_memory_stream_chat.call_args.kwargs
            assert call_kwargs["pp_reranker_id"] == "reranker-1"
            assert call_kwargs["pp_llm_id"] == "llm-1"
            assert call_kwargs["pp_relevance_threshold"] == 0.5
            assert call_kwargs["pp_llm_temp"] == 0.7
            assert call_kwargs["pp_chronological_resort"] is True

    def test_metadata_filter_applied_to_space_key(self):
        with patch("goodmem_deepeval.client.ApiClient"):
            client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="key")
            client._stream_client.retrieve_memory_stream_chat = MagicMock(return_value=iter([]))

            client.retrieve_memories(
                query="test",
                spaces=["sp-1"],
                metadata_filters={"filter": "CAST(val('$.source') AS TEXT) = 'docs'"},
            )

            call_kwargs = client._stream_client.retrieve_memory_stream_chat.call_args.kwargs
            space_keys = call_kwargs["space_keys"]
            assert len(space_keys) == 1
            assert space_keys[0].filter == "CAST(val('$.source') AS TEXT) = 'docs'"

    def test_parses_events_into_results(self):
        with patch("goodmem_deepeval.client.ApiClient"):
            client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="key")

            mock_chunk = MagicMock()
            mock_chunk.memory_id = "mem-1"
            mock_chunk.space_id = "sp-1"
            mock_chunk.relevance_score = 0.9
            mock_inner = MagicMock()
            mock_inner.chunkText = "hello world"
            mock_chunk.chunk = mock_inner

            mock_item = MagicMock()
            mock_item.chunk = mock_chunk

            mock_event = MagicMock()
            mock_event.retrieved_item = mock_item
            mock_event.memory_definition = MagicMock(metadata={"source": "test"})

            client._stream_client.retrieve_memory_stream_chat = MagicMock(return_value=iter([mock_event]))

            result = client.retrieve_memories(query="q", spaces=["sp-1"])

            assert len(result["results"]) == 1
            assert result["results"][0]["content"] == "hello world"
            assert result["results"][0]["score"] == 0.9


class TestListAndDelete:
    def test_list_spaces(self):
        with patch("goodmem_deepeval.client.ApiClient"):
            client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="key")
            mock_resp = MagicMock()
            mock_resp.spaces = [MagicMock(space_id="sp-1")]
            client._spaces_api.list_spaces = MagicMock(return_value=mock_resp)

            result = client.list_spaces()
            assert len(result) == 1

    def test_get_memory(self):
        with patch("goodmem_deepeval.client.ApiClient"):
            client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="key")
            client._memories_api.get_memory = MagicMock(return_value=MagicMock(memory_id="mem-1"))

            result = client.get_memory("mem-1", include_content=True)
            assert result.memory_id == "mem-1"
            client._memories_api.get_memory.assert_called_once_with(
                id="mem-1", include_content=True, include_processing_history=False
            )

    def test_delete_memory(self):
        with patch("goodmem_deepeval.client.ApiClient"):
            client = GoodMemEvalClient(base_url="http://localhost:8080", api_key="key")
            client._memories_api.delete_memory = MagicMock()

            client.delete_memory("mem-1")
            client._memories_api.delete_memory.assert_called_once_with(id="mem-1")
