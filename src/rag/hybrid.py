"""
src/rag/hybrid.py

Stage 4.6 of the RAG pipeline: hybrid retrieval (dense + BM25, fused with RRF).

WHY RRF AND NOT WEIGHTED-SCORE FUSION
-------------------------------------
Cosine scores live in ~[0,1]; BM25 scores are unbounded and query-dependent.
Adding them directly lets whichever scale is bigger dominate, and normalizing
two incompatible scales is fiddly. RRF sidesteps the problem: it uses only
rank position. Each chunk gets 1/(k + rank) from each list it appears in,
summed. k=60 is the Cormack et al. default and the standard across Elastic,
Weaviate, and Qdrant.

SECTION FILTER
--------------
Dense supports a section filter via Qdrant; BM25 doesn't know about sections.
We apply the filter *post-fusion* — fuse first, then drop results whose
section doesn't match. Slightly wasteful (BM25 may rank out-of-section
chunks) but correct, and cheap at this corpus size.
"""

from __future__ import annotations

from rag.bm25_index import BM25Hit, BM25Index
from rag.retriever import DenseRetriever
from rag.store import SearchResult


def reciprocal_rank_fusion(
    dense_hits: list[SearchResult],
    bm25_hits: list[BM25Hit],
    k: int = 60,
) -> list[SearchResult]:
    """Fuse two ranked lists by RRF. Returns SearchResults ordered by fused score.

    Uses RANK position (1-indexed), not raw score. This is the central RRF
    idea and the reason fusion works across incompatible score scales.
    """
    fused_scores: dict[str, float] = {}
    payloads: dict[str, SearchResult] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        payloads[hit.chunk_id] = hit

    for rank, hit in enumerate(bm25_hits, start=1):
        fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)

    results: list[SearchResult] = []
    for chunk_id, fused_score in fused_scores.items():
        if chunk_id not in payloads:
            continue  # BM25-only hits without dense payload: v1 limitation
        original = payloads[chunk_id]
        results.append(
            SearchResult(
                chunk_id=original.chunk_id,
                text=original.text,
                source=original.source,
                title=original.title,
                section=original.section,
                score=fused_score,  # RRF score — different scale from cosine
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results


class HybridRetriever:
    """Drop-in replacement for DenseRetriever. Same string-in/results-out contract."""

    def __init__(self, dense: DenseRetriever, bm25: BM25Index) -> None:
        self.dense = dense
        self.bm25 = bm25

    def search(
        self,
        query: str,
        top_k: int = 5,
        section: str | None = None,
        fusion_k: int = 30,
    ) -> list[SearchResult]:
        # Pull a generous pool from each retriever. Section is applied
        # post-fusion below so it works uniformly regardless of which
        # retriever surfaced the chunk.
        dense_hits = self.dense.search(query, top_k=fusion_k, section=None)
        bm25_hits = self.bm25.search(query, k=fusion_k)

        fused = reciprocal_rank_fusion(dense_hits, bm25_hits)

        if section is not None:
            fused = [r for r in fused if r.section == section]

        return fused[:top_k]


if __name__ == "__main__":
    from rag.bm25_index import BM25Index
    from rag.chunker import chunk_documents
    from rag.embedder import Embedder
    from rag.loader import load_documents
    from rag.store import VectorStore

    print("Loading corpus...")
    docs = load_documents("data/raw/fastapi/docs/en/docs")
    chunks = chunk_documents(docs)

    print("Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(chunks)

    print("Connecting to Qdrant + loading embedder...")
    store = VectorStore()
    embedder = Embedder()
    dense = DenseRetriever(store=store, embedder=embedder)
    hybrid = HybridRetriever(dense=dense, bm25=bm25)

    queries = [
        "OAuth2PasswordBearer",
        "how do I protect my API from anonymous users",
        "query parameters",
    ]
    for q in queries:
        print(f"\n=== Q: {q} ===")
        print("  Dense-only:")
        for r in dense.search(q, top_k=5):
            print(f"    [{r.score:.3f}] {r.chunk_id}")
        print("  Hybrid (RRF):")
        for r in hybrid.search(q, top_k=5):
            print(f"    [{r.score:.4f}] {r.chunk_id}")