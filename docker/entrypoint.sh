#!/usr/bin/env bash
# Dispatch on the first argument so one image plays three roles.
# Usage inside a compose service:
#   command: api      -> run the FastAPI app on 0.0.0.0:8000
#   command: ui       -> run the Streamlit UI on 0.0.0.0:8501
#   command: indexer  -> one-shot: load -> chunk -> embed -> upsert into Qdrant

set -euo pipefail

ROLE="${1:-api}"

case "$ROLE" in
    api)
        exec uvicorn rag.api:app --host 0.0.0.0 --port 8000
        ;;
    ui)
        # Streamlit's default port is 8501. --server.address binds outside
        # the container; --server.headless skips the welcome prompt.
        exec streamlit run src/rag/ui.py \
            --server.address 0.0.0.0 \
            --server.port 8501 \
            --server.headless true
        ;;
    indexer)
        # One-shot population of Qdrant. Exits when done.
        # src/rag/store.py's __main__ does load -> chunk -> embed -> upsert.
        exec python -m rag.store
        ;;
    *)
        echo "Unknown role: $ROLE (expected: api | ui | indexer)" >&2
        exit 1
        ;;
esac