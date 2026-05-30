"""
src/rag/api.py

HTTP service layer over the RAG pipeline.

Endpoints:
    POST /ask     — main Q&A
    GET  /health  — liveness + indexed chunk count

Heavy state (embedder, vector store, generator, bm25 index, retrievers,
reranker) is loaded ONCE at app startup via a lifespan handler and shared
across requests.

Retriever selection:
    Env var RETRIEVER=dense|hybrid|reranked sets the default (default: dense).
    Per-request override: AskRequest.retriever overrides for that call.

Run locally:
    uv run uvicorn rag.api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.bm25_index import BM25Index
from rag.embedder import Embedder
from rag.generator import Generator
from rag.hybrid import HybridRetriever
from rag.reranker import CrossEncoderReranker, RerankingRetriever
from rag.retriever import DenseRetriever
from rag.store import VectorStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag.api")


RetrieverName = Literal["dense", "hybrid", "reranked"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(4, ge=1, le=20)
    section: str | None = Field(None)
    retriever: RetrieverName | None = Field(None)


class Source(BaseModel):
    chunk_id: str
    title: str
    source: str
    section: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    model: str
    retriever: RetrieverName
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    default_retriever: RetrieverName
    retrievers_available: list[RetrieverName]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading embedder, vector store, generator, BM25, reranker...")
    t0 = time.perf_counter()

    embedder = Embedder()
    store = VectorStore()
    generator = Generator()

    bm25: BM25Index | None = BM25Index()
    try:
        bm25.build_from_qdrant(store)
        logger.info(f"BM25 index built from Qdrant: {len(bm25._chunk_ids)} chunks")
    except Exception as exc:
        logger.warning(f"BM25 build_from_qdrant failed: {exc}. Hybrid/reranked disabled.")
        bm25 = None

    dense_retriever = DenseRetriever(store=store, embedder=embedder)
    hybrid_retriever = (
        HybridRetriever(dense=dense_retriever, bm25=bm25) if bm25 is not None else None
    )

    # Reranker. Lazy-loads the cross-encoder on first .score() call so
    # startup stays fast; first reranked request pays the model-load cost.
    reranker = CrossEncoderReranker()
    reranked_retriever = (
        RerankingRetriever(inner=hybrid_retriever, reranker=reranker)
        if hybrid_retriever is not None
        else None
    )

    default = os.getenv("RETRIEVER", "dense").lower()
    if default not in ("dense", "hybrid", "reranked"):
        logger.warning(f"Unknown RETRIEVER={default!r}, falling back to 'dense'.")
        default = "dense"

    retrievers: dict[str, object] = {"dense": dense_retriever}
    if hybrid_retriever is not None:
        retrievers["hybrid"] = hybrid_retriever
    if reranked_retriever is not None:
        retrievers["reranked"] = reranked_retriever

    if default not in retrievers:
        logger.warning(
            f"RETRIEVER={default!r} requested but unavailable; using 'dense'."
        )
        default = "dense"

    app.state.embedder = embedder
    app.state.store = store
    app.state.generator = generator
    app.state.retrievers = retrievers
    app.state.default_retriever = default

    try:
        n = store.count()
        logger.info(f"Vector store ready: {n} chunks indexed.")
        app.state.chunks_indexed = n
    except Exception as exc:
        logger.error(f"Vector store probe failed: {exc}")
        app.state.chunks_indexed = 0

    logger.info(
        f"Startup complete in {time.perf_counter() - t0:.2f}s "
        f"(default retriever: {default}, available: {list(retrievers)})"
    )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="docs-rag",
    description=(
        "A retrieval-augmented Q&A service over the FastAPI documentation. "
        "Supports dense, hybrid (BM25 + dense + RRF), and reranked "
        "(hybrid + cross-encoder) retrieval; selectable via the RETRIEVER "
        "env var or per-request."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        chunks_indexed=getattr(app.state, "chunks_indexed", 0),
        default_retriever=getattr(app.state, "default_retriever", "dense"),
        retrievers_available=list(getattr(app.state, "retrievers", {}).keys()),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()

    section = req.section
    if section in (None, "", "string"):
        section = None

    chosen = req.retriever or app.state.default_retriever
    if chosen not in app.state.retrievers:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Retriever {chosen!r} not available. "
                f"Available: {list(app.state.retrievers)}"
            ),
        )
    retriever = app.state.retrievers[chosen]
    logger.info(
        f"/ask question={req.question!r} top_k={req.top_k} retriever={chosen}"
    )

    try:
        results = retriever.search(
            req.question, top_k=req.top_k, section=section,
        )
        if not results:
            if section is not None:
                msg = (
                    f"No chunks matched the filter section={section!r}. "
                    f"Try removing the section filter or using one of: "
                    f"tutorial, advanced, deployment, how-to, learn, "
                    f"reference, about."
                )
            else:
                msg = "No results found. Vector store may be empty."
            raise HTTPException(status_code=404, detail=msg)

        answer = app.state.generator.generate(req.question, results)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pipeline failure")
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = int((time.perf_counter() - t0) * 1000)

    return AskResponse(
        question=req.question,
        answer=answer.text,
        sources=[
            Source(
                chunk_id=r.chunk_id,
                title=r.title,
                source=r.source,
                section=r.section,
                score=round(r.score, 4),
                snippet=(
                    r.text.split("\n\n", 1)[-1][:280] + "..."
                    if len(r.text) > 280
                    else r.text.split("\n\n", 1)[-1]
                ),
            )
            for r in results
        ],
        model=answer.model,
        retriever=chosen,
        latency_ms=latency_ms,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
    )