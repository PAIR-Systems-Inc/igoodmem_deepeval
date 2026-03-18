"""
Dataset loading and caching for scaled RAG benchmarks.

Provides a standard schema (BenchmarkDocument, BenchmarkQuery, BenchmarkDataset)
and loaders for real-world datasets like SQuAD 2.0.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from urllib.request import urlretrieve

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ── Standard schema ──────────────────────────────────────────────────────────


@dataclass
class BenchmarkDocument:
    """A single document to be indexed by all providers."""

    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkQuery:
    """A single evaluation query, optionally with a gold answer."""

    query: str
    expected_answer: Optional[str] = None
    source_doc_id: Optional[str] = None


@dataclass
class BenchmarkDataset:
    """A complete benchmark dataset: documents + evaluation queries."""

    name: str
    documents: List[BenchmarkDocument] = field(default_factory=list)
    queries: List[BenchmarkQuery] = field(default_factory=list)

    def summary(self) -> str:
        total_chars = sum(len(d.text) for d in self.documents)
        approx_pages = total_chars / 3000  # ~3000 chars per page
        return (
            f"Dataset '{self.name}': {len(self.documents)} documents "
            f"(~{approx_pages:.0f} pages), {len(self.queries)} queries"
        )


# ── SQuAD 2.0 loader ────────────────────────────────────────────────────────


def _download_squad(cache_dir: Path) -> Path:
    """Download SQuAD 2.0 training set if not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "train-v2.0.json"
    if path.exists():
        print(f"  📦 Using cached SQuAD data: {path}")
        return path
    print(f"  ⬇️  Downloading SQuAD 2.0 (~40MB)...")
    urlretrieve(SQUAD_URL, path)
    print(f"  ✅ Saved to {path}")
    return path


def _make_doc_id(title: str) -> str:
    """Create a stable, filesystem-safe doc ID from an article title."""
    return hashlib.md5(title.encode()).hexdigest()[:12]


def load_squad_subset(
    num_articles: int = 200,
    queries_per_article: int = 2,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> BenchmarkDataset:
    """
    Load a subset of SQuAD 2.0 as a BenchmarkDataset.

    Each SQuAD article becomes a document; answerable questions become queries
    with gold answers.

    Args:
        num_articles: Number of Wikipedia articles to include (~1.5 pages each).
        queries_per_article: Number of Q&A pairs to sample per article.
        seed: Random seed for reproducible selection.
        cache_dir: Where to cache the downloaded JSON.

    Returns:
        BenchmarkDataset with documents and queries.
    """
    cache_dir = cache_dir or DATA_DIR
    squad_path = _download_squad(cache_dir)

    print(f"  📖 Parsing SQuAD 2.0...")
    with open(squad_path) as f:
        raw = json.load(f)

    rng = random.Random(seed)

    # Each SQuAD "data" entry is a Wikipedia article with multiple paragraphs
    all_articles = raw["data"]

    # Filter to articles that have at least queries_per_article answerable questions
    eligible = []
    for article in all_articles:
        # Collect all paragraphs as the full document text
        paragraphs = [p["context"] for p in article["paragraphs"]]
        full_text = "\n\n".join(paragraphs)

        # Collect all answerable questions across all paragraphs
        answerable_qas = []
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                if not qa.get("is_impossible", False) and qa.get("answers"):
                    answerable_qas.append(qa)

        if len(answerable_qas) >= queries_per_article and len(full_text) > 200:
            eligible.append(
                {
                    "title": article["title"],
                    "text": full_text,
                    "qas": answerable_qas,
                }
            )

    if len(eligible) < num_articles:
        print(
            f"  ⚠️  Only {len(eligible)} eligible articles found "
            f"(requested {num_articles})"
        )
        num_articles = len(eligible)

    selected = rng.sample(eligible, num_articles)

    documents: List[BenchmarkDocument] = []
    queries: List[BenchmarkQuery] = []

    for article in selected:
        doc_id = _make_doc_id(article["title"])
        documents.append(
            BenchmarkDocument(
                doc_id=doc_id,
                text=article["text"],
                metadata={
                    "source": "squad",
                    "title": article["title"],
                },
            )
        )

        sampled_qas = rng.sample(article["qas"], queries_per_article)
        for qa in sampled_qas:
            # Take the first (shortest) gold answer
            gold_answer = min(
                (a["text"] for a in qa["answers"]),
                key=len,
            )
            queries.append(
                BenchmarkQuery(
                    query=qa["question"],
                    expected_answer=gold_answer,
                    source_doc_id=doc_id,
                )
            )

    dataset = BenchmarkDataset(
        name=f"squad-{num_articles}",
        documents=documents,
        queries=queries,
    )
    print(f"  ✅ {dataset.summary()}")
    return dataset
