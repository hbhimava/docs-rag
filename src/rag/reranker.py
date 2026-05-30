"""
src/rag/reranker.py

Stage 4.7 of the RAG pipeline: cross-encoder reranking.

WHY THIS EXISTS
---------------
Dense (bi-encoder) and BM25 score chunks against queries WITHOUT ever
reading them together: dense compares two independently-produced vectors,
BM25 counts words. Neither model gets to attend to the query and the
chunk jointly. A cross-encoder does — it concatenates [query, chunk] and
runs a transformer over the pair, so every attention layer sees both
sides at once. This lets it recognize semantic relevance that survives
zero literal-word overlap (e.g. "protect API from anonymous users" vs a
chunk titled "Security - First Steps" that never says "anonymous" or
"protect").

The cost: ~50ms per pair on CPU (vs free vector math for dense). So we
don't rerank the whole corpus — we let hybrid retrieve a small candidate
pool (~20) and rerank just those. Recall comes from hybrid; precision
comes from the reranker. This is the canonical retrieve-then-rerank
architecture.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag.store import SearchResult


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    """Thin wrapper around a sentence-transformers CrossEncoder.

    First call lazily loads the model (~280MB download on first run).
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _ensure_loaded(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """Rescore candidates by cross-encoder relevance and re-sort.

        Returns a new list of SearchResult ordered by reranker score
        (descending). The .score field is overwritten with the
        reranker's score, which is on yet another scale (typically
        unbounded logits, often negative). Don't mix it with cosine
        or RRF scores in displays.
        """
        if not candidates:
            return []

        model = self._ensure_loaded()
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)  # one logit per pair

        rescored: list[SearchResult] = []
        for c, s in zip(candidates, scores):
            rescored.append(
                SearchResult(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source=c.source,
                    title=c.title,
                    section=c.section,
                    score=float(s),
                )
            )
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored


class RerankingRetriever:
    """Wraps another retriever; pulls a generous candidate pool, then reranks.

    Same `search(query, top_k, section) -> list[SearchResult]` contract
    as DenseRetriever and HybridRetriever, so it's a drop-in for the
    generator/API.
    """

    def __init__(
        self,
        inner,                       # any object with .search(query, top_k, section)
        reranker: CrossEncoderReranker,
        candidate_pool: int = 20,
    ) -> None:
        self.inner = inner
        self.reranker = reranker
        self.candidate_pool = candidate_pool

    def search(
        self,
        query: str,
        top_k: int = 5,
        section: str | None = None,
    ) -> list[SearchResult]:
        # Recall stage: pull a pool larger than top_k so the reranker has
        # something to choose from. The section filter is applied by the
        # inner retriever, so out-of-section chunks are gone by the time
        # we rerank — no wasted cross-encoder calls.
        candidates = self.inner.search(
            query, top_k=self.candidate_pool, section=section
        )
        if not candidates:
            return []

        # Precision stage: rescore the small pool with the cross-encoder.
        reranked = self.reranker.score(query, candidates)
        return reranked[:top_k]


if __name__ == "__main__":
    # Smoke test: hybrid candidates -> reranked top-5 for the three
    # queries we've been using. The "anonymous users" query is the
    # interesting one — see if the security chunks float up.
    from rag.bm25_index import BM25Index
    from rag.embedder import Embedder
    from rag.hybrid import HybridRetriever
    from rag.retriever import DenseRetriever
    from rag.store import VectorStore

    print("Loading store + embedder...")
    store = VectorStore()
    embedder = Embedder()

    print("Building BM25 from Qdrant...")
    bm25 = BM25Index()
    bm25.build_from_qdrant(store)

    dense = DenseRetriever(store=store, embedder=embedder)
    hybrid = HybridRetriever(dense=dense, bm25=bm25)

    print("Loading cross-encoder (first run downloads ~280MB)...")
    reranker = CrossEncoderReranker()
    reranked_retriever = RerankingRetriever(inner=hybrid, reranker=reranker)

    queries = [
        "OAuth2PasswordBearer",
        "dependency injection in FastAPI",
        "query parameters",
    ]
    for q in queries:
        print(f"\n=== Q: {q} ===")
        print("  Hybrid (RRF):")
        for r in hybrid.search(q, top_k=5):
            print(f"    [{r.score:.4f}] {r.chunk_id}")
        print("  Hybrid + reranker:")
        for r in reranked_retriever.search(q, top_k=5):
            print(f"    [{r.score:7.3f}] {r.chunk_id}")