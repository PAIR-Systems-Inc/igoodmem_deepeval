"""Contextual AI retrieval provider for the benchmark framework."""
from __future__ import annotations

import os
import tempfile
import time
from typing import List, Optional, Tuple

from .base import Generator, RAGPipeline, RetrievalProvider
from ..datasets import BenchmarkDocument

DATASTORE_PREFIX = "scaled-benchmark"
DOC_POLL_INTERVAL = 3  # seconds between status checks
DOC_POLL_TIMEOUT = 300  # max seconds to wait for all docs to finish ingesting


class _ContextualAIPipeline:
    """Wraps Contextual AI agent query API into a RAGPipeline callable."""

    def __init__(self, client, agent_id: str, generator: Generator, top_k: int):
        self._client = client
        self._agent_id = agent_id
        self._generator = generator
        self._top_k = top_k

    def __call__(self, query: str) -> Tuple[str, List[str]]:
        # Use retrievals_only to skip their LLM — we use our shared generator
        response = self._client.agents.query.create(
            agent_id=self._agent_id,
            messages=[{"role": "user", "content": query}],
            stream=False,
            retrievals_only=True,
            include_retrieval_content_text=True,
        )

        retrieval_context = []
        if hasattr(response, "retrieval_contents") and response.retrieval_contents:
            for content in response.retrieval_contents[: self._top_k]:
                text = getattr(content, "content_text", None) or ""
                if text:
                    retrieval_context.append(text)

        answer = self._generator(query, retrieval_context)
        return answer, retrieval_context


class ContextualAIProvider(RetrievalProvider):
    """Benchmarks Contextual AI retrieval.

    Note: Contextual AI handles its own document parsing and chunking.
    Unlike other providers, we cannot control chunk size — so this is
    a system-level comparison rather than a controlled chunking comparison.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("CONTEXTUAL_AI_API_KEY")
        if not self._api_key:
            raise RuntimeError("CONTEXTUAL_AI_API_KEY is required.")
        self._client = None
        self._datastore_id = None
        self._agent_id = None

    @property
    def name(self) -> str:
        return "ContextualAI"

    def _find_existing_datastore(self, target_name: str) -> Optional[str]:
        """Check if a datastore with the given name already exists and has documents."""
        try:
            # datastores.list() returns a paginated SyncDatastoresPage
            for ds in self._client.datastores.list():
                if ds.name == target_name:
                    # Check if it has documents
                    docs_page = self._client.datastores.documents.list(
                        datastore_id=ds.id, limit=1
                    )
                    for doc in docs_page:
                        return ds.id  # Has at least one doc
            return None
        except Exception:
            return None

    def _find_existing_agent(self, datastore_id: str, agent_name_prefix: str) -> Optional[str]:
        """Check if a benchmark agent already exists by name convention."""
        try:
            # Agent model only has id/name/description — no datastore_ids.
            # Match by naming convention instead.
            for agent in self._client.agents.list():
                if agent.name and agent.name.startswith("Benchmark Agent"):
                    # Use datastores.list(agent_id=...) to verify linkage
                    try:
                        for ds in self._client.datastores.list(agent_id=agent.id):
                            if ds.id == datastore_id:
                                return agent.id
                    except Exception:
                        pass
            return None
        except Exception:
            return None

    def setup(self, documents: List[BenchmarkDocument]) -> None:
        from contextual import ContextualAI

        self._client = ContextualAI(api_key=self._api_key)

        datastore_name = f"{DATASTORE_PREFIX}-{len(documents)}"

        # Check for existing datastore with data
        existing_ds_id = self._find_existing_datastore(datastore_name)
        if existing_ds_id:
            self._datastore_id = existing_ds_id
            print(f"  📂 Reusing existing datastore: {datastore_name}")

            # Find or create agent for this datastore
            existing_agent = self._find_existing_agent(self._datastore_id, "Benchmark Agent")
            if existing_agent:
                self._agent_id = existing_agent
                print(f"  🤖 Reusing existing agent: {self._agent_id}")
                return
            else:
                agent = self._client.agents.create(
                    name=f"Benchmark Agent ({len(documents)} docs)",
                    description="Auto-created by scaled RAG benchmark",
                    datastore_ids=[self._datastore_id],
                )
                self._agent_id = agent.id
                print(f"  🤖 Created agent: {self._agent_id}")
                return

        # Create new datastore
        datastore = self._client.datastores.create(name=datastore_name)
        self._datastore_id = datastore.id
        print(f"  📂 Created datastore: {datastore_name} (id: {self._datastore_id})")

        # Upload documents as temp text files
        doc_ids = []
        total = len(documents)
        for i, doc in enumerate(documents):
            title = doc.metadata.get("title", doc.doc_id)
            try:
                # Write text to a temp file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", prefix=f"bench_{doc.doc_id}_", delete=False
                ) as tmp:
                    tmp.write(doc.text)
                    tmp_path = tmp.name

                # Upload the file
                with open(tmp_path, "rb") as f:
                    result = self._client.datastores.documents.ingest(
                        datastore_id=self._datastore_id, file=f
                    )
                    doc_ids.append(result.id)

                # Clean up temp file
                os.unlink(tmp_path)

            except Exception as e:
                print(f"  ⚠️  Failed to upload '{title}': {e}")
                # Clean up temp file on error
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"  📝 Uploaded {i + 1}/{total} documents")

        # Poll for all documents to finish ingesting
        print(f"  ⏳ Waiting for {len(doc_ids)} documents to finish processing...")
        start_wait = time.time()
        pending = set(doc_ids)
        completed = 0
        failed = 0

        while pending and (time.time() - start_wait) < DOC_POLL_TIMEOUT:
            still_pending = set()
            for doc_id in pending:
                try:
                    meta = self._client.datastores.documents.metadata(
                        doc_id, datastore_id=self._datastore_id
                    )
                    status = meta.status if hasattr(meta, "status") else "unknown"
                    if status == "completed":
                        completed += 1
                    elif status == "failed":
                        failed += 1
                        print(f"  ⚠️  Document {doc_id} failed processing")
                    else:
                        still_pending.add(doc_id)
                except Exception:
                    still_pending.add(doc_id)
            pending = still_pending
            if pending:
                elapsed = time.time() - start_wait
                print(f"  ⏳ {completed} completed, {len(pending)} still processing... ({elapsed:.0f}s)")
                time.sleep(DOC_POLL_INTERVAL)

        if pending:
            print(f"  ⚠️  {len(pending)} documents still processing after {DOC_POLL_TIMEOUT}s timeout")

        print(f"  ✅ Ingestion done: {completed} completed, {failed} failed, {len(pending)} timed out")

        # Create agent linked to this datastore
        print("  🤖 Creating agent...")
        agent = self._client.agents.create(
            name=f"Benchmark Agent ({len(documents)} docs)",
            description="Auto-created by scaled RAG benchmark",
            datastore_ids=[self._datastore_id],
        )
        self._agent_id = agent.id
        print(f"  🤖 Agent created: {self._agent_id}")

        # Brief wait for agent to be ready
        time.sleep(5)
        print("  ✅ Contextual AI ready")

    def make_pipeline(
        self,
        generator: Generator,
        top_k: int,
        metadata_filter: Optional[str] = None,
    ) -> RAGPipeline:
        if not self._client or not self._agent_id:
            raise RuntimeError("Call setup() before make_pipeline().")
        return _ContextualAIPipeline(
            client=self._client,
            agent_id=self._agent_id,
            generator=generator,
            top_k=top_k,
        )

    def teardown(self) -> None:
        # Don't delete datastore/agent — they can be reused on next run
        pass
