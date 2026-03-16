---
name: GoodMem Integration in Deepeval
description: GoodMem vector memory API integration as a knowledge base client under deepeval/integrations/goodmem/, tested and working
type: project
---

GoodMem integration was added to deepeval as a knowledge base client at `deepeval/integrations/goodmem/`.

**Why:** GoodMem provides vector-based memory storage and semantic retrieval, which maps naturally to deepeval's RAG evaluation workflow (retrieval_context in LLMTestCase).

**How to apply:** When working on GoodMem-related changes in deepeval, the integration lives at `deepeval/integrations/goodmem/` with three files: `__init__.py`, `client.py`, `types.py`. Tests are at `tests/test_integrations/test_goodmem_e2e.py` (standalone) and `tests/test_integrations/test_goodmem.py` (pytest). The OpenAI Ada embedder on the GoodMem server fails; use the Voyage embedder (019c2aff-c470-778d-9f4e-89a74c77890f) via GOODMEM_EMBEDDER_ID env var.
