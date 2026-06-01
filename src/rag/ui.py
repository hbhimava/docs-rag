"""
src/rag/ui.py

Streamlit chat UI for docs-rag.

A thin client over the FastAPI backend at /ask. Owns no model state
and runs no RAG logic itself -- this is by design so the UI can be
swapped (or removed) without touching the pipeline.

Run locally (the FastAPI server must be running on port 8000):

    uv run streamlit run src/rag/ui.py
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
DEFAULT_TOP_K = 4
REQUEST_TIMEOUT_S = 60


st.set_page_config(
    page_title="docs-rag · FastAPI Q&A",
    page_icon="📚",
    layout="centered",
)

st.title("📚 docs-rag")
st.caption(
    "Ask anything about the FastAPI docs. Answers are grounded in "
    "retrieved documentation chunks with inline `[n]` citations."
)


# Cache the health probe for 60s so we don't hammer /health on every
# Streamlit rerun (which happens on every widget interaction).
@st.cache_data(ttl=60)
def fetch_health() -> dict | None:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except Exception:
        return None


health = fetch_health()


with st.sidebar:
    st.header("Settings")

    top_k = st.slider(
        "Chunks to retrieve (top_k)",
        min_value=1, max_value=10, value=DEFAULT_TOP_K,
        help=(
            "How many chunks to retrieve before generation. Higher = "
            "more context (and cost), but past ~6 the LLM tends to "
            "lose focus."
        ),
    )

    # Retriever toggle. Options come from /health so the UI only shows
    # retrievers the API actually has loaded. "(server default)" lets
    # the server's RETRIEVER env var decide -- useful for A/B by env.
    if health is not None:
        available = health.get("retrievers_available", ["dense"])
        default_retriever = health.get("default_retriever", "dense")
    else:
        available = ["dense"]
        default_retriever = "dense"

    retriever_options = [f"(server default: {default_retriever})"] + available
    retriever_choice = st.radio(
        "Retriever",
        options=retriever_options,
        index=0,
        help=(
            "Which retrieval pipeline to use.\n\n"
            "- **dense**: BGE embeddings + cosine. Fast, semantic.\n"
            "- **hybrid**: dense + BM25 lexical, fused with RRF. "
            "Helps on rare-identifier queries.\n"
            "- **reranked**: hybrid + cross-encoder rerank. Slower "
            "(~8s on CPU); see README's eval section for tradeoffs."
        ),
    )

    section = st.selectbox(
        "Restrict to section (optional)",
        options=[
            "(no filter)",
            "tutorial",
            "advanced",
            "deployment",
            "how-to",
            "learn",
            "reference",
            "about",
        ],
        help=(
            "Filter retrieval to one top-level docs section. "
            "Useful for narrowing scope."
        ),
    )

    st.divider()

    st.subheader("Backend status")
    if health is not None:
        st.success(f"✅ API up · {health['chunks_indexed']:,} chunks indexed")
        st.caption(
            f"Default retriever: `{default_retriever}` · "
            f"Available: {', '.join(f'`{r}`' for r in available)}"
        )
    else:
        st.error(f"❌ API unreachable at {API_URL}")

    st.caption(f"API: `{API_URL}`")


if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        sources = msg.get("sources")
        if sources:
            with st.expander(f"📎 Sources ({len(sources)})"):
                for i, s in enumerate(sources, start=1):
                    st.markdown(
                        f"**[{i}] {s['title']}** · "
                        f"`{s['source']}` · "
                        f"section `{s['section']}` · "
                        f"score {s['score']:.3f}"
                    )
                    st.caption(s["snippet"])

        meta = msg.get("meta")
        if meta:
            # Show the retriever that actually served this query --
            # important when the user picks "(server default)" or when
            # the server overrides their pick (e.g. unavailable option).
            retriever_label = (
                f" · retriever: `{meta['retriever']}`"
                if meta.get("retriever") else ""
            )
            st.caption(
                f"⏱ {meta['latency_ms']} ms{retriever_label} · "
                f"{meta['model']} · "
                f"tokens: {meta['prompt_tokens']} in / "
                f"{meta['completion_tokens']} out"
            )


for m in st.session_state.messages:
    render_message(m)


question = st.chat_input("Ask a question about the FastAPI docs...")

if question:
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            payload: dict = {
                "question": question,
                "top_k": top_k,
            }
            if section != "(no filter)":
                payload["section"] = section
            # Only set retriever if user picked a specific one --
            # absent means "let the server default decide".
            if not retriever_choice.startswith("(server default"):
                payload["retriever"] = retriever_choice

            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    f"{API_URL}/ask",
                    json=payload,
                    timeout=REQUEST_TIMEOUT_S,
                )
                client_latency = int((time.perf_counter() - t0) * 1000)
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
                st.stop()

        if resp.status_code == 200:
            data = resp.json()
            assistant_msg = {
                "role": "assistant",
                "content": data["answer"],
                "sources": data["sources"],
                "meta": {
                    "latency_ms": data["latency_ms"],
                    "model": data["model"],
                    "retriever": data.get("retriever"),
                    "prompt_tokens": data["prompt_tokens"],
                    "completion_tokens": data["completion_tokens"],
                    "client_latency_ms": client_latency,
                },
            }
        else:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            assistant_msg = {
                "role": "assistant",
                "content": f"❌ API returned **{resp.status_code}**: {detail}",
            }

        st.session_state.messages.append(assistant_msg)
        render_message(assistant_msg)