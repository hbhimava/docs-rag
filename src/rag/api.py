"""
src/rag/api.py

HTTP service layer over the RAG pipeline.

One endpoint that matters (POST /ask) and one operational endpoint
(GET /health). Heavy state (embedder, vector store, generator) is
loaded ONCE at app startup via a lifespan handler and shared across
all requests.

Run locally:
    uv run uvicorn rag.api:app --reload --host 0.0.0.0 --port 8000

Then visit:
    http://localhost:8000/docs   <- interactive Swagger UI
    http://localhost:8000/health <- json status
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.embedder import Embedder
from rag.generator import Generator
from rag.store import VectorStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag.api")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(4, ge=1, le=20)
    section: str | None = Field(None)


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
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading embedder, vector store, and generator...")
    t0 = time.perf_counter()

    app.state.embedder = Embedder()
    app.state.store = VectorStore()
    app.state.generator = Generator()

    try:
        n = app.state.store.count()
        logger.info(f"Vector store ready: {n} chunks indexed.")
        app.state.chunks_indexed = n
    except Exception as exc:
        logger.error(f"Vector store probe failed: {exc}")
        app.state.chunks_indexed = 0

    logger.info(f"Startup complete in {time.perf_counter() - t0:.2f}s")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="docs-rag",
    description=(
        "A retrieval-augmented Q&A service over the FastAPI "
        "documentation. Built from first principles with hybrid "
        "search and reranking (added in later blocks)."
    ),
    version="0.1.0",
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
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    logger.info(f"/ask question={req.question!r} top_k={req.top_k}")

    # Sanitize the section filter. Swagger UI prefills optional string
    # fields with the literal placeholder "string"; empty strings are
    # also treated as "no filter" rather than a filter that matches the
    # empty section.
    section = req.section
    if section in (None, "", "string"):
        section = None

    try:
        query_vec = app.state.embedder.embed_query(req.question)
        results = app.state.store.search(
            query_vec, top_k=req.top_k, section=section,
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
        latency_ms=latency_ms,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
    )
