# GoodMem Integration for DeepEval

GoodMem is a semantic memory server. This integration lets you store documents, retrieve relevant chunks via semantic search, and feed them directly into DeepEval's evaluation framework.

## Installation

GoodMem requires a running GoodMem server. No additional Python packages are needed — the integration uses only `requests`, which is already a DeepEval dependency.

```bash
pip install deepeval
```

## Quick Start

```python
from deepeval.integrations.goodmem import GoodMemClient
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric, FaithfulnessMetric

# 1. Connect
client = GoodMemClient(
    base_url="http://localhost:8080",
    api_key="your-api-key",
)

# 2. Discover available embedders
embedders = client.list_embedders()
embedder_id = embedders[0].embedder_id

# 3. Create a space
space = client.create_space(name="my-knowledge-base", embedder_id=embedder_id)

# 4. Store documents
client.create_memory(space_id=space.space_id, text_content="Paris is the capital of France.")
client.create_memory(space_id=space.space_id, file_path="/path/to/document.pdf")

# 5. Retrieve context and evaluate
query = "What is the capital of France?"
context = client.retrieve_as_context(query=query, space_ids=space.space_id)

test_case = LLMTestCase(
    input=query,
    actual_output="The capital of France is Paris.",
    retrieval_context=context,
)

evaluate([test_case], [ContextualRelevancyMetric(), FaithfulnessMetric()])
```

## Environment Variables

Set these to avoid hardcoding credentials:

```bash
export GOODMEM_BASE_URL="http://localhost:8080"
export GOODMEM_API_KEY="your-api-key"
```

```python
import os
client = GoodMemClient(
    base_url=os.environ["GOODMEM_BASE_URL"],
    api_key=os.environ["GOODMEM_API_KEY"],
)
```

---

## API Reference

### `GoodMemClient`

```python
GoodMemClient(base_url: str, api_key: str)
```

Main client for all GoodMem operations.

---

#### Model Discovery

```python
client.list_embedders() -> List[GoodMemEmbedder]
```
Returns all embedding models available on the server.

```python
client.list_rerankers() -> List[GoodMemReranker]
```
Returns all reranking models available on the server.

```python
client.list_llms() -> List[GoodMemLLM]
```
Returns all LLM models available on the server (used for LLM-assisted retrieval).

```python
client.list_spaces() -> List[Dict[str, Any]]
```
Returns all memory spaces as raw dicts.

---

#### Space Management

```python
client.create_space(
    name: str,
    embedder_id: str,
    chunk_size: int = 256,
    chunk_overlap: int = 25,
    keep_strategy: str = "KEEP_END",
    length_measurement: str = "CHARACTER_COUNT",
) -> GoodMemSpace
```

Creates a new memory space. If a space with the same name already exists, it is reused instead of creating a duplicate.

| Parameter | Description |
|---|---|
| `name` | Display name for the space |
| `embedder_id` | ID of the embedding model to use (from `list_embedders()`) |
| `chunk_size` | Target chunk size in characters (default: 256) |
| `chunk_overlap` | Overlap between adjacent chunks (default: 25) |
| `keep_strategy` | How to handle chunks at document boundaries: `"KEEP_END"` or `"KEEP_START"` |
| `length_measurement` | How chunk size is measured: `"CHARACTER_COUNT"` or `"TOKEN_COUNT"` |

---

#### Memory Management

```python
client.create_memory(
    space_id: str,
    text_content: Optional[str] = None,
    file_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> GoodMemMemory
```

Stores a document as a memory in the given space. Provide either `text_content` or `file_path`, not both.

| Parameter | Description |
|---|---|
| `space_id` | Target space ID (from `create_space()`) |
| `text_content` | Plain text to store |
| `file_path` | Path to a local file (PDF, DOCX, TXT, CSV, images, etc.) |
| `metadata` | Optional key-value metadata attached to the memory |

Supported file types: PDF, PNG, JPG, GIF, WEBP, TXT, HTML, Markdown, CSV, JSON, XML, DOC, DOCX, XLS, XLSX, PPT, PPTX.

Memory processing is asynchronous — the returned object has `processing_status: "PENDING"`. Use `retrieve_memories(wait_for_indexing=True)` to block until indexing completes.

```python
client.get_memory(
    memory_id: str,
    include_content: bool = True,
) -> Dict[str, Any]
```

Fetches a specific memory by ID. Returns a dict with a `"memory"` key and optionally a `"content"` key.

```python
client.delete_memory(memory_id: str) -> Dict[str, Any]
```

Permanently deletes a memory and all its associated chunks and vectors.

---

#### Retrieval

```python
client.retrieve_memories(
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
) -> GoodMemRetrievalResult
```

Performs semantic similarity search across one or more spaces.

| Parameter | Description |
|---|---|
| `query` | Natural language search query |
| `space_ids` | Single space ID or list of space IDs to search |
| `max_results` | Maximum number of chunks to return (default: 5) |
| `include_memory_definition` | Include full memory metadata in results |
| `wait_for_indexing` | Poll up to 60 seconds if memories are still processing |
| `reranker_id` | Optional reranker model ID for improved ranking (from `list_rerankers()`) |
| `llm_id` | Optional LLM ID to generate an abstract reply from results |
| `relevance_threshold` | Minimum relevance score (0.0–1.0) to include a chunk |
| `llm_temperature` | Temperature for LLM-assisted summarization |
| `chronological_resort` | Re-sort results chronologically after retrieval |

```python
client.retrieve_as_context(
    query: str,
    space_ids: Union[str, List[str]],
    max_results: int = 5,
    wait_for_indexing: bool = True,
    **kwargs,
) -> List[str]
```

Convenience wrapper around `retrieve_memories()` that returns chunk texts as `List[str]`, ready for use as `LLMTestCase(retrieval_context=...)`. Accepts the same keyword arguments as `retrieve_memories()`.

---

## Data Types

### `GoodMemSpace`

```python
@dataclass
class GoodMemSpace:
    space_id: str
    name: str
    embedder_id: Optional[str]
    reused: bool                          # True if existing space was reused
    chunking_config: Optional[Dict]
```

### `GoodMemMemory`

```python
@dataclass
class GoodMemMemory:
    memory_id: str
    space_id: str
    processing_status: str               # "PENDING" | "PROCESSING" | "READY"
    content_type: Optional[str]          # MIME type, e.g. "text/plain"
    file_name: Optional[str]
```

### `GoodMemRetrievalResult`

```python
@dataclass
class GoodMemRetrievalResult:
    result_set_id: str
    results: List[GoodMemRetrievedChunk]
    memories: List[Dict[str, Any]]
    total_results: int
    query: str
    abstract_reply: Optional[Dict]       # Present when llm_id is set
```

### `GoodMemRetrievedChunk`

```python
@dataclass
class GoodMemRetrievedChunk:
    chunk_id: Optional[str]
    chunk_text: Optional[str]
    memory_id: Optional[str]
    relevance_score: Optional[float]
    memory_index: Optional[int]
```

### `GoodMemEmbedder` / `GoodMemReranker` / `GoodMemLLM`

```python
@dataclass
class GoodMemEmbedder:
    embedder_id: str
    display_name: Optional[str]
    model_identifier: Optional[str]      # e.g. "text-embedding-3-small"
```

---

## Usage Patterns

### Retrieval-Only Evaluation

Score retrieval quality without an LLM generator using DeepEval's `ContextualRelevancyMetric`:

```python
from deepeval.integrations.goodmem import GoodMemClient
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

client = GoodMemClient(base_url="...", api_key="...")

queries = [
    "What is the capital of France?",
    "Who invented the telephone?",
]

test_cases = [
    LLMTestCase(
        input=q,
        actual_output="",  # not evaluated for retrieval-only
        retrieval_context=client.retrieve_as_context(q, space_ids="your-space-id"),
    )
    for q in queries
]

evaluate(test_cases, [ContextualRelevancyMetric()])
```

### Full RAG Pipeline Evaluation

```python
import openai
from deepeval.integrations.goodmem import GoodMemClient
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric

client = GoodMemClient(base_url="...", api_key="...")

def generate(query: str, context: list[str]) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only the provided context.\n\n" + "\n".join(context)},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content

queries = ["What is the capital of France?"]

test_cases = []
for query in queries:
    context = client.retrieve_as_context(query, space_ids="your-space-id")
    answer = generate(query, context)
    test_cases.append(LLMTestCase(input=query, actual_output=answer, retrieval_context=context))

evaluate(test_cases, [
    AnswerRelevancyMetric(),
    FaithfulnessMetric(),
    ContextualRelevancyMetric(),
])
```

### Reranking

```python
rerankers = client.list_rerankers()
reranker_id = rerankers[0].reranker_id

context = client.retrieve_as_context(
    query="What is the capital of France?",
    space_ids="your-space-id",
    max_results=10,
    reranker_id=reranker_id,
)
```

### Multi-Space Search

```python
context = client.retrieve_as_context(
    query="...",
    space_ids=["space-id-1", "space-id-2", "space-id-3"],
    max_results=5,
)
```

### LLM-Assisted Retrieval (Abstract Reply)

```python
llms = client.list_llms()
llm_id = llms[0].llm_id

result = client.retrieve_memories(
    query="Summarize the key findings",
    space_ids="your-space-id",
    llm_id=llm_id,
    llm_temperature=0.2,
)

# result.abstract_reply contains the LLM-generated summary
```

### Relevance Filtering

```python
context = client.retrieve_as_context(
    query="...",
    space_ids="your-space-id",
    max_results=10,
    relevance_threshold=0.7,   # only chunks with score >= 0.7
)
```

---

## Running the Tests

Integration tests require a live GoodMem server:

```bash
export GOODMEM_BASE_URL="http://localhost:8080"
export GOODMEM_API_KEY="your-api-key"
export GOODMEM_EMBEDDER_ID="optional-embedder-id"   # auto-detected if omitted
export GOODMEM_PDF_PATH="/path/to/test.pdf"          # optional, skips PDF test if unset

pytest tests/test_integrations/test_goodmem.py -v
```

Alternatively, run the standalone e2e script which does not require the full pytest setup:

```bash
python tests/test_integrations/test_goodmem_e2e.py
```

---

## DeepEval Metrics Reference

| Metric | What it measures | Requires generator? |
|---|---|---|
| `ContextualRelevancyMetric` | Are retrieved chunks relevant to the query? | No |
| `ContextualPrecisionMetric` | Are the most relevant chunks ranked first? | No |
| `ContextualRecallMetric` | Does retrieved context cover the expected answer? | Yes (expected_output) |
| `FaithfulnessMetric` | Is the answer grounded in the retrieved context? | Yes |
| `AnswerRelevancyMetric` | Is the answer relevant to the query? | Yes |
| `HallucinationMetric` | Does the answer contradict the retrieved context? | Yes |
