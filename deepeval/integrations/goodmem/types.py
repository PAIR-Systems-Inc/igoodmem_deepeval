from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GoodMemSpace:
    space_id: str
    name: str
    embedder_id: Optional[str] = None
    reused: bool = False
    chunking_config: Optional[Dict[str, Any]] = None


@dataclass
class GoodMemMemory:
    memory_id: str
    space_id: str
    processing_status: str = "PENDING"
    content_type: Optional[str] = None
    file_name: Optional[str] = None


@dataclass
class GoodMemRetrievedChunk:
    chunk_id: Optional[str] = None
    chunk_text: Optional[str] = None
    memory_id: Optional[str] = None
    relevance_score: Optional[float] = None
    memory_index: Optional[int] = None


@dataclass
class GoodMemRetrievalResult:
    result_set_id: str = ""
    results: List[GoodMemRetrievedChunk] = field(default_factory=list)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    total_results: int = 0
    query: str = ""
    abstract_reply: Optional[Dict[str, Any]] = None


@dataclass
class GoodMemEmbedder:
    embedder_id: str
    display_name: Optional[str] = None
    model_identifier: Optional[str] = None


@dataclass
class GoodMemReranker:
    reranker_id: str
    display_name: Optional[str] = None
    model_identifier: Optional[str] = None


@dataclass
class GoodMemLLM:
    llm_id: str
    display_name: Optional[str] = None
    model_identifier: Optional[str] = None
