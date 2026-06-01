"""
src/rag/store.py

Stage 4 of the RAG pipeline: storing and searching vectors.
Wraps a Qdrant collection.

Uses the modern qdrant-client API:
  - create_collection / delete_collection (replaces recreate_collection)
  - query_points (replaces search)
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rag.chunker import Chunk
from rag.embedder import EMBEDDING_DIM


@dataclass
class SearchResult:
    """One retrieved chunk plus its similarity score."""
    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    score: float


class VectorStore:
    """Opinionated wrapper around a single Qdrant collection."""

    

    def __init__(
    self,
    url: str | None = None,
    collection: str = "fastapi_docs",) -> None:
         self.client = QdrantClient(url=url or os.getenv("QDRANT_URL", "http://localhost:6333"))
         self.collection = collection
    ...

    def recreate_collection(self) -> None:
        """Delete the collection if it exists, then create it fresh.

        Idempotent: safe to call repeatedly while iterating on chunks
        or embeddings. Uses the modern split API instead of the
        deprecated recreate_collection() method.
        """
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors,                 # np.ndarray, shape (len(chunks), DIM)
        batch_size: int = 128,
    ) -> None:
        """Insert chunks + their vectors into the collection in batches."""
        points: list[PointStruct] = []
        for i, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "source": chunk.source,
                        "title": chunk.title,
                        "section": chunk.section,
                        "header": chunk.metadata.get("header", ""),
                    },
                )
            )

        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            self.client.upsert(
                collection_name=self.collection,
                points=batch,
            )

    def search(
        self,
        query_vector,            # np.ndarray, shape (DIM,)
        top_k: int = 5,
        section: str | None = None,
    ) -> list[SearchResult]:
        """Return the top_k most similar chunks to query_vector.

        Uses the modern query_points() API. The response object wraps
        the hit list in a .points attribute, hence the unwrapping below.
        """
        query_filter = None
        if section is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for h in response.points:
            p = h.payload or {}
            results.append(
                SearchResult(
                    chunk_id=p.get("chunk_id", ""),
                    text=p.get("text", ""),
                    source=p.get("source", ""),
                    title=p.get("title", ""),
                    section=p.get("section", ""),
                    score=h.score,
                )
            )
        return results

    def count(self) -> int:
        """How many points are currently in the collection."""
        return self.client.count(
            collection_name=self.collection
        ).count


if __name__ == "__main__":
    from rag.loader import load_documents
    from rag.chunker import chunk_documents
    from rag.embedder import Embedder

    print("Loading documents...")
    docs = load_documents("data/raw/fastapi/docs/en/docs")

    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"  {len(chunks)} chunks")

    print("Embedding (this runs the model over every chunk)...")
    embedder = Embedder()
    texts = [c.text for c in chunks]
    vectors = embedder.embed_documents(texts)
    print(f"  vectors: {vectors.shape}")

    print("Indexing into Qdrant...")
    store = VectorStore()
    store.recreate_collection()
    store.upsert_chunks(chunks, vectors)
    print(f"  points in collection: {store.count()}")

    print("\n--- Test searches ---")
    test_queries = [
        "How do I declare query parameters?",
        "How do I run FastAPI in production?",
        "How do I handle errors and return custom status codes?",
    ]
    for q in test_queries:
        qv = embedder.embed_query(q)
        results = store.search(qv, top_k=3)
        print(f"\nQ: {q}")
        for r in results:
            print(f"  [{r.score:.3f}] {r.title}  ({r.source})")
