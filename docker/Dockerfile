# Single image used for three roles: api, ui, indexer.
# Docker compose runs it with different commands. Same dependencies for
# all three (the UI doesn't strictly need ML libs, but maintaining two
# Dockerfiles isn't worth the disk savings at this scale).

FROM python:3.11-slim

# System deps: curl for healthchecks, build-essential for any wheels that
# don't have prebuilt linux/aarch64 binaries (sentence-transformers
# sometimes pulls one).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use uv inside the container too — matches the dev workflow.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first so the layer cache survives code edits.
COPY pyproject.toml uv.lock* ./

# Install into the system Python (no venv inside container — pointless layer).
RUN uv pip install --system --no-cache .

# Copy the application code and corpus.
COPY src ./src
COPY eval ./eval
COPY data ./data
COPY README.md ./

# Pre-download the BGE embedder and the cross-encoder reranker so the
# first request after `docker compose up` doesn't stall on a 400MB
# HuggingFace pull. Cache them to a path that ships in the image.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-base'); \
print('models pre-downloaded')"

# Make `rag` importable.
ENV PYTHONPATH=/app/src

# Entrypoint script dispatches on the first arg: api | ui | indexer
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["api"]