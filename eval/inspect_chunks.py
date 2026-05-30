"""
eval/inspect_chunks.py

Dump the text of specific chunks so you can decide which siblings
legitimately answer a question. Use this to widen expected_chunk_ids
in eval/dataset.json.

Usage:
    uv run python eval/inspect_chunks.py \\
        tutorial/response-model.md::5 \\
        tutorial/response-model.md::6 \\
        tutorial/response-model.md::7 \\
        tutorial/response-model.md::8
"""

from __future__ import annotations

import sys

from rag.store import VectorStore


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: inspect_chunks.py CHUNK_ID [CHUNK_ID ...]")
        sys.exit(1)

    target = set(sys.argv[1:])
    store = VectorStore()

    found: dict[str, str] = {}
    offset = None
    while True:
        pts, offset = store.client.scroll(
            collection_name=store.collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break
        for p in pts:
            cid = (p.payload or {}).get("chunk_id", "")
            if cid in target:
                found[cid] = (p.payload or {}).get("text", "")
        if offset is None:
            break

    for cid in sys.argv[1:]:
        print(f"\n{'=' * 70}")
        print(f"  {cid}")
        print("=" * 70)
        text = found.get(cid)
        if text is None:
            print("  <not found in Qdrant>")
        else:
            print(text[:800])
            if len(text) > 800:
                print(f"\n  ...(truncated, {len(text)} chars total)")


if __name__ == "__main__":
    main()