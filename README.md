# docs-rag

A production-style RAG (Retrieval-Augmented Generation) pipeline that answers
questions about the [FastAPI](https://fastapi.tiangolo.com/) documentation.

**Status:** 🚧 in active development (build-out weekend May 16-18, 2026)

## What this is

An end-to-end RAG system built from first principles — no LangChain or
LlamaIndex framework — to make every pipeline stage understandable and
tunable. Indexes the FastAPI English documentation (~1,360 chunks across
144 files), answers questions with inline citations grounded in retrieved
context.

## Pipeline stages

1. **Loader** — walks the docs tree, strips markdown noise (frontmatter,
   admonition delimiters, rendered-CLI box-drawing art), preserves
   structural metadata.
2. **Chunker** — markdown-aware recursive splitting on H2-H4 headers with
   ~1000-char target and ~150-char overlap; each chunk carries a
   title + heading prefix so the embedding encodes its topic.
3. **Embedder** — `BAAI/bge-small-en-v1.5` via `sentence-transformers`,
   384-dim, L2-normalized. Respects BGE's document/query asymmetry
   (queries get the instruction prefix; documents don't).
4. **Vector store** — Qdrant collection with cosine distance, payload
   filtering on metadata fields (source, section).
5. **Generator** — `gpt-4o-mini` with a grounding-enforced system prompt
   and numbered-context citations.

## Stack

Python 3.11 · uv · sentence-transformers · Qdrant · OpenAI · FastAPI ·
Streamlit · Docker · Hugging Face Spaces

## Planned (Day 2-3)

- Hybrid retrieval (BM25 sparse + dense, RRF fusion)
- Cross-encoder reranking (`BAAI/bge-reranker-base`)
- Evaluation harness (RAGAS metrics on a hand-curated eval set)
- FastAPI HTTP layer + Streamlit chat UI
- Dockerized deploy on Hugging Face Spaces
- GitHub Actions CI (lint, test, eval-on-push)

## Run locally

Requirements: Docker, Python 3.11, an OpenAI API key.

```bash
# 1. Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install deps and set up secrets
uv sync
cp .env.example .env  # then edit .env and add your OPENAI_API_KEY

# 3. Get the FastAPI docs as our corpus
mkdir -p data/raw
cd data/raw
git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git
cd fastapi && git sparse-checkout set docs/en/docs && cd ../..

# 4. Start the vector store
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

# 5. Index and run an end-to-end test
uv run python src/rag/store.py        # builds the index
uv run python src/rag/generator.py    # asks a few sample questions
```

## License

MIT — see [LICENSE](LICENSE).
