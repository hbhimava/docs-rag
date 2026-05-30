"""
eval/run_eval.py

Run retrieval eval across all three retrievers (dense, hybrid, reranked).

For each test case:
  - Run each retriever and get its top-k chunk_ids.
  - Compute hit@k (any expected chunk in top-k?) and MRR
    (1 / rank of first expected chunk, or 0 if none found).

Outputs:
  - Summary table to stdout: hit@k and MRR per retriever.
  - eval/results.csv: per-query results across all retrievers, so you
    can sort by failures and dig in.

Run:
    uv run python eval/run_eval.py \\
        --dataset eval/dataset.json \\
        --top-k 5

Dataset format (a JSON array):
    [
      {
        "question": "...",
        "expected_chunk_ids": ["...", "..."],
        "tags": ["..."]   // optional
      },
      ...
    ]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from rag.bm25_index import BM25Index
from rag.embedder import Embedder
from rag.hybrid import HybridRetriever
from rag.reranker import CrossEncoderReranker, RerankingRetriever
from rag.retriever import DenseRetriever
from rag.store import VectorStore


def hit_at_k(retrieved_ids: list[str], expected: set[str], k: int) -> int:
    return int(any(cid in expected for cid in retrieved_ids[:k]))


def reciprocal_rank(retrieved_ids: list[str], expected: set[str], k: int) -> float:
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in expected:
            return 1.0 / i
    return 0.0


def evaluate_retriever(
    name: str,
    retriever,
    dataset: list[dict],
    top_k: int,
) -> tuple[dict, list[dict]]:
    """Run one retriever across the dataset. Return aggregate + per-query rows."""
    hits: list[int] = []
    rrs: list[float] = []
    latencies: list[float] = []
    rows: list[dict] = []

    for i, case in enumerate(dataset, start=1):
        question = case["question"]
        expected = set(case["expected_chunk_ids"])

        t0 = time.perf_counter()
        results = retriever.search(question, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_ids = [r.chunk_id for r in results]
        h = hit_at_k(retrieved_ids, expected, top_k)
        rr = reciprocal_rank(retrieved_ids, expected, top_k)

        hits.append(h)
        rrs.append(rr)
        latencies.append(latency_ms)

        rows.append({
            "retriever": name,
            "question": question,
            "expected_chunk_ids": "|".join(sorted(expected)),
            "retrieved_top1": retrieved_ids[0] if retrieved_ids else "",
            "retrieved_topk": "|".join(retrieved_ids[:top_k]),
            "hit_at_k": h,
            "reciprocal_rank": round(rr, 4),
            "latency_ms": round(latency_ms, 1),
        })

        print(f"  [{i:3d}/{len(dataset)}] hit={h} rr={rr:.3f} ({latency_ms:5.0f}ms)  {question[:60]}")

    aggregate = {
        "retriever": name,
        "n": len(dataset),
        f"hit_at_{top_k}": round(statistics.mean(hits), 4),
        "mrr": round(statistics.mean(rrs), 4),
        "latency_ms_p50": round(statistics.median(latencies), 1),
        "latency_ms_mean": round(statistics.mean(latencies), 1),
    }
    return aggregate, rows


def print_summary(aggregates: list[dict], top_k: int) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    hdr = f"{'retriever':<12} {'n':>4} {'hit@'+str(top_k):>8} {'MRR':>7} {'p50_ms':>8} {'mean_ms':>9}"
    print(hdr)
    print("-" * 78)
    for a in aggregates:
        print(
            f"{a['retriever']:<12} {a['n']:>4} "
            f"{a[f'hit_at_{top_k}']:>8.4f} {a['mrr']:>7.4f} "
            f"{a['latency_ms_p50']:>8.1f} {a['latency_ms_mean']:>9.1f}"
        )
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/dataset.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--results-csv", default="eval/results.csv")
    parser.add_argument(
        "--skip-reranker", action="store_true",
        help="Skip the reranker (slow). Useful while iterating.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(
            f"Dataset not found: {dataset_path}\n"
            "Run eval/generate_dataset.py first, then curate to eval/dataset.json."
        )

    with dataset_path.open() as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} test cases from {dataset_path}")

    # Same wiring as the API — keep them consistent so eval reflects prod.
    print("Building retrievers...")
    store = VectorStore()
    embedder = Embedder()
    dense = DenseRetriever(store=store, embedder=embedder)

    bm25 = BM25Index()
    bm25.build_from_qdrant(store)
    hybrid = HybridRetriever(dense=dense, bm25=bm25)

    retrievers: list[tuple[str, object]] = [
        ("dense", dense),
        ("hybrid", hybrid),
    ]
    if not args.skip_reranker:
        reranker = CrossEncoderReranker()
        reranked = RerankingRetriever(inner=hybrid, reranker=reranker)
        retrievers.append(("reranked", reranked))

    aggregates: list[dict] = []
    all_rows: list[dict] = []
    for name, r in retrievers:
        print(f"\n--- Evaluating: {name} ---")
        agg, rows = evaluate_retriever(name, r, dataset, top_k=args.top_k)
        aggregates.append(agg)
        all_rows.extend(rows)

    print_summary(aggregates, args.top_k)

    out = Path(args.results_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nPer-query results written to {out}")
    print(
        "Sort by reciprocal_rank ascending to find the worst cases per "
        "retriever — those are where the interesting failure analysis "
        "lives."
    )


if __name__ == "__main__":
    main()