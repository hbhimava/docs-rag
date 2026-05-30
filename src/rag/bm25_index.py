"""
src/rag/bm25_index.py

Stage 4.5 of the RAG pipeline: a sparse (lexical) retriever to sit
alongside the dense Qdrant retriever.

WHY THIS EXISTS
---------------
Dense (BGE + Qdrant cosine) finds chunks by *meaning* but compresses each
into 384 numbers, so exact tokens (`OAuth2PasswordBearer`, `@app.get`,
error codes) blur into their semantic neighborhood. BM25 keeps a literal
word index and scores by:
  - TF (saturating): more occurrences -> higher, with diminishing returns
  - IDF: rare words across the corpus count far more than common ones
         (this is why stopwords drop out naturally — "the" is everywhere)
  - Length normalization: tight relevant paragraph can beat a long
         rambling page

bm25s implements all three; we feed it well-tokenized text.

KEYING
------
SearchResult.chunk_id is the SHARED key across dense and BM25. BM25's
internal positions are an implementation detail; we translate them back
to chunk_ids on the way out so RRF can fuse the two lists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import bm25s

from rag.chunker import Chunk


# Identifier-aware tokenizer. Default English tokenizers split on every
# non-letter and often stem, which DESTROYS code identifiers — exactly the
# tokens BM25 exists to catch. We lowercase and keep runs of [a-z0-9_]
# together, no stemming, no stopwords (IDF handles common words).
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Hit:
    chunk_id: str
    score: float


class BM25Index:
    def __init__(self) -> None:
        self._retriever: bm25s.BM25 | None = None
        self._chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Build from Chunk objects (used during indexing pipeline)."""
        self._chunk_ids = [c.chunk_id for c in chunks]
        corpus_tokens = [tokenize(c.text) for c in chunks]
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens)

    def build_from_qdrant(self, store) -> None:
        """Build from chunks already in Qdrant. Used at API startup so we
        don't have to re-chunk the corpus on every restart.

        Pulls all points (chunk_id + text payload) via scroll and rebuilds
        in-process. At ~1,360 chunks this takes well under a second.
        """
        chunk_ids: list[str] = []
        texts: list[str] = []

        offset = None
        while True:
            points, offset = store.client.scroll(
                collection_name=store.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for p in points:
                payload = p.payload or {}
                chunk_ids.append(payload.get("chunk_id", str(p.id)))
                texts.append(payload.get("text", ""))
            if offset is None:
                break

        self._chunk_ids = chunk_ids
        self._retriever = bm25s.BM25()
        self._retriever.index([tokenize(t) for t in texts])

    def search(self, query: str, k: int = 30) -> list[BM25Hit]:
        if self._retriever is None:
            raise RuntimeError("BM25Index.search called before build/load")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        k = min(k, len(self._chunk_ids))
        indices, scores = self._retriever.retrieve([query_tokens], k=k)

        return [
            BM25Hit(chunk_id=self._chunk_ids[int(pos)], score=float(score))
            for pos, score in zip(indices[0], scores[0])
        ]

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(p / "bm25"))
        (p / "chunk_ids.json").write_text(json.dumps(self._chunk_ids))

    def load(self, path: str) -> None:
        p = Path(path)
        self._retriever = bm25s.BM25.load(str(p / "bm25"))
        self._chunk_ids = json.loads((p / "chunk_ids.json").read_text())


if __name__ == "__main__":
    from rag.chunker import chunk_documents
    from rag.loader import load_documents

    docs = load_documents("data/raw/fastapi/docs/en/docs")
    chunks = chunk_documents(docs)
    print(f"{len(chunks)} chunks")

    index = BM25Index()
    index.build(chunks)

    for q in [
        "OAuth2PasswordBearer",
        "how do I protect my API from anonymous users",
        "query parameters",
    ]:
        print(f"\nQ: {q}")
        for hit in index.search(q, k=3):
            print(f"  [{hit.score:6.2f}] {hit.chunk_id}")