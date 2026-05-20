"""
src/rag/embedder.py

Stage 3 of the RAG pipeline: turning text into vectors.

Wraps a sentence-transformers model (BAAI/bge-small-en-v1.5).

Key correctness detail: BGE models are asymmetric.
  - DOCUMENTS / chunks are embedded as-is.
  - QUERIES must be prefixed with a specific instruction string,
    or retrieval quality silently degrades.
This module makes that asymmetry impossible to get wrong by exposing
two explicit methods: embed_documents() and embed_query().
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 384


class Embedder:
    """Thin, opinionated wrapper around a SentenceTransformer."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed chunk texts (no instruction prefix). Shape: (n, 384)."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one query (WITH BGE instruction prefix). Shape: (384,)."""
        prefixed = QUERY_INSTRUCTION + query
        vec = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec[0]


if __name__ == "__main__":
    emb = Embedder()

    docs = [
        "FastAPI lets you declare query parameters with type hints.",
        "To deploy FastAPI you can use Uvicorn behind a process manager.",
        "Bananas are a good source of potassium.",
    ]
    doc_vecs = emb.embed_documents(docs)

    query = "How do I add query parameters in FastAPI?"
    q_vec = emb.embed_query(query)

    print(f"\nModel: {emb.model_name}")
    print(f"Doc vectors shape:   {doc_vecs.shape}")
    print(f"Query vector shape:  {q_vec.shape}")

    sims = doc_vecs @ q_vec
    print(f"\nQuery: {query!r}\n")
    for text, score in zip(docs, sims):
        print(f"  {score:+.3f}  {text}")

    best = int(np.argmax(sims))
    print(f"\nBest match (index {best}): {docs[best]}")
    print("Expected: index 0 (the query-parameters sentence).")
