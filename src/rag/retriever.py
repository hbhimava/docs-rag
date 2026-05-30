"""
src/rag/retriever.py

Stage 4.4 of the RAG pipeline: a thin wrapper that gives the dense path
the same `search(query: str, ...)` contract as HybridRetriever.

VectorStore stays as the pure Qdrant data layer (vector in, results out).
DenseRetriever owns the embedder and exposes string-in/results-out so the
API doesn't have to orchestrate embedding itself.
"""

from __future__ import annotations

from rag.embedder import Embedder
from rag.store import SearchResult, VectorStore


class DenseRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 5,
        section: str | None = None,
    ) -> list[SearchResult]:
        qv = self.embedder.embed_query(query)
        return self.store.search(qv, top_k=top_k, section=section)