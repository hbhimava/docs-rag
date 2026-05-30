"""
eval/generate_dataset.py

Bootstraps an eval dataset by asking gpt-4o-mini to produce one realistic
question per chunk, sampled from your corpus. Writes a JSONL of candidates
for you to curate by hand.

Chunks are pulled from Qdrant (not re-chunked from disk), so the dataset
can only reference chunks that actually exist in the live retrieval system.
This avoids silent drift between the eval set and what the retrievers see.

The output is INTENTIONALLY noisy — the model will produce some questions
phrased in the chunk's own vocabulary (which would inflate retrieval
scores). Your curation step is where the eval gets its quality:
  - Delete questions that are too obvious or that quote the chunk directly.
  - Rephrase a handful to use user vocabulary that doesn't appear in the
    chunk (e.g. "protect from anonymous users" instead of "implement
    OAuth2"). These are the queries that actually stress the system.
  - Add 5-10 hard cases yourself.

Run:
    uv run python eval/generate_dataset.py \\
        --sample 60 \\
        --out eval/dataset_candidates.jsonl

Then curate down to ~25-30 high-quality cases and save as eval/dataset.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag.store import VectorStore


load_dotenv()


PROMPT = """\
You are helping build an evaluation set for a documentation search system.

Given the following documentation chunk, write ONE realistic question that
a developer might type into a search box, where THIS chunk would be a good
answer.

Rules:
- Phrase it as a developer would — natural, sometimes vague, sometimes
  using different vocabulary than the docs.
- Don't quote the chunk verbatim. Don't include rare exact tokens from
  the chunk unless a real user would know them.
- 5-20 words. No quotation marks. No leading "How do I" if it sounds
  unnatural.
- If the chunk is reference material (like a list of class signatures
  with no real prose), respond with exactly: SKIP

Chunk:
\"\"\"
{chunk_text}
\"\"\"

Question:"""


def fetch_all_chunks(store: VectorStore) -> list[dict]:
    """Pull every chunk's payload out of Qdrant via scroll.

    Same access pattern BM25Index uses at startup, so 'what chunks exist'
    has one source of truth across the system.
    """
    payloads: list[dict] = []
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
            if p.payload:
                payloads.append(p.payload)
        if offset is None:
            break
    return payloads


def generate_question(client: OpenAI, chunk_text: str) -> str | None:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[{"role": "user", "content": PROMPT.format(chunk_text=chunk_text)}],
    )
    q = response.choices[0].message.content.strip().strip('"').strip()
    if q.upper().startswith("SKIP"):
        return None
    return q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample", type=int, default=60,
        help="How many chunks to sample (we'll generate one question each).",
    )
    parser.add_argument(
        "--out", default="eval/dataset_candidates.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Pulling chunks from Qdrant...")
    store = VectorStore()
    chunks = fetch_all_chunks(store)
    print(f"  {len(chunks)} chunks in store")

    if not chunks:
        raise SystemExit(
            "No chunks in Qdrant — did you run the indexing pipeline? "
            "(uv run python src/rag/store.py)"
        )

    random.seed(args.seed)
    sampled = random.sample(chunks, min(args.sample, len(chunks)))
    print(f"  sampled {len(sampled)} chunks")

    client = OpenAI()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with out_path.open("w") as f:
        for i, chunk in enumerate(sampled, start=1):
            try:
                q = generate_question(client, chunk.get("text", ""))
            except Exception as exc:
                print(f"  [{i}] error: {exc}")
                continue

            if q is None:
                skipped += 1
                continue

            record = {
                "question": q,
                "expected_chunk_ids": [chunk.get("chunk_id", "")],
                "source": chunk.get("source", ""),
                "section": chunk.get("section", ""),
                "title": chunk.get("title", ""),
                "tags": [chunk.get("section", "")],
            }
            f.write(json.dumps(record) + "\n")
            written += 1
            print(f"  [{i:3d}] {chunk.get('chunk_id', '')}")
            print(f"        Q: {q}")

    print(
        f"\nWrote {written} candidates to {out_path} "
        f"(skipped {skipped} reference-style chunks)."
    )
    print(
        "\nNEXT: open the JSONL, curate to ~25-30 questions, add 5-10 hard\n"
        "user-vocabulary queries by hand, and save as eval/dataset.json\n"
        "(a JSON array of the same shape)."
    )


if __name__ == "__main__":
    main()