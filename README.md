docs-rag

A retrieval-augmented Q&A system over the FastAPI documentation, built from first principles. Hybrid retrieval (BM25 + dense embeddings + RRF), cross-encoder reranking, and a real eval harness — measured, not asserted.

This project exists for two reasons. First, to actually understand each piece of a RAG pipeline by building it without leaning on a framework — no LangChain, no LlamaIndex, just the underlying libraries and the data flowing through them. Second, to test the standard production recipes (hybrid search, retrieve-then-rerank) on a real corpus and report what actually moved the needle and what didn't.
The system answers natural-language questions about FastAPI by retrieving documentation chunks from a vector store, fusing them with a sparse lexical retriever, optionally reranking them with a cross-encoder, then asking gpt-4o-mini to write a grounded answer with inline [n] citations. Three retrievers are selectable at runtime — dense, hybrid, or reranked — both by environment variable and per-request. An eval harness measures hit-rate and MRR across all three.
The interesting part is in EVALUATION.md (or section 5 below if you're reading this top to bottom): on this corpus, the simplest retriever wins. Hybrid is a wash. Reranking actively hurts at 200× the latency. The why — and what to do about it — is the real point of the project.

Quickstart
Three commands. First time takes ~10 minutes (the image build pre-downloads the BGE embedder and cross-encoder reranker so the first request isn't slow):
bash# 1. Build the image and bring up Qdrant
docker compose build
docker compose up -d qdrant

# 2. Populate the vector store (one-shot; data persists in ./qdrant_storage)
docker compose run --rm indexer

# 3. Bring up the API and UI
docker compose up -d api ui
Then:

Streamlit UI: http://localhost:8501
FastAPI Swagger: http://localhost:8000/docs
Health check: http://localhost:8000/health

All services bind to 127.0.0.1 only — not exposed to your LAN. Stop everything with docker compose down.
You'll need an .env file at the repo root with an OpenAI key:
OPENAI_API_KEY=sk-...

What it works like
Ask a question. The system retrieves relevant chunks, generates an answer grounded in those chunks, and shows the answer with [n] citations linking back to the source chunks. Both the chosen retriever and per-query latency are visible.
Show Image
The same flow is available over HTTP:
bashcurl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I declare query parameters?", "top_k": 4, "retriever": "hybrid"}'
Response includes the generated answer, the source chunks (with their chunk_id for traceability), the chosen retriever, latency, and token usage.

Pipeline
#mermaid-r1ae-r44{font-family:"Anthropic Sans",system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:16px;fill:#191919;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#mermaid-r1ae-r44 .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#mermaid-r1ae-r44 .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#mermaid-r1ae-r44 .error-icon{fill:#CC785C;}#mermaid-r1ae-r44 .error-text{fill:#3387a3;stroke:#3387a3;}#mermaid-r1ae-r44 .edge-thickness-normal{stroke-width:1px;}#mermaid-r1ae-r44 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-r1ae-r44 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-r1ae-r44 .edge-thickness-invisible{stroke-width:0;fill:none;}#mermaid-r1ae-r44 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-r1ae-r44 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-r1ae-r44 .marker{fill:#91918D;stroke:#91918D;}#mermaid-r1ae-r44 .marker.cross{stroke:#91918D;}#mermaid-r1ae-r44 svg{font-family:"Anthropic Sans",system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:16px;}#mermaid-r1ae-r44 p{margin:0;}#mermaid-r1ae-r44 .label{font-family:"Anthropic Sans",system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#191919;}#mermaid-r1ae-r44 .cluster-label text{fill:#3387a3;}#mermaid-r1ae-r44 .cluster-label span{color:#3387a3;}#mermaid-r1ae-r44 .cluster-label span p{background-color:transparent;}#mermaid-r1ae-r44 .label text,#mermaid-r1ae-r44 span{fill:#191919;color:#191919;}#mermaid-r1ae-r44 .node rect,#mermaid-r1ae-r44 .node circle,#mermaid-r1ae-r44 .node ellipse,#mermaid-r1ae-r44 .node polygon,#mermaid-r1ae-r44 .node path{fill:#F0F0EB;stroke:#D9D8D5;stroke-width:1px;}#mermaid-r1ae-r44 .rough-node .label text,#mermaid-r1ae-r44 .node .label text,#mermaid-r1ae-r44 .image-shape .label,#mermaid-r1ae-r44 .icon-shape .label{text-anchor:middle;}#mermaid-r1ae-r44 .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#mermaid-r1ae-r44 .rough-node .label,#mermaid-r1ae-r44 .node .label,#mermaid-r1ae-r44 .image-shape .label,#mermaid-r1ae-r44 .icon-shape .label{text-align:center;}#mermaid-r1ae-r44 .node.clickable{cursor:pointer;}#mermaid-r1ae-r44 .root .anchor path{fill:#91918D!important;stroke-width:0;stroke:#91918D;}#mermaid-r1ae-r44 .arrowheadPath{fill:#0b0b0b;}#mermaid-r1ae-r44 .edgePath .path{stroke:#91918D;stroke-width:1px;}#mermaid-r1ae-r44 .flowchart-link{stroke:#91918D;fill:none;}#mermaid-r1ae-r44 .edgeLabel{background-color:#F5E6D8;text-align:center;}#mermaid-r1ae-r44 .edgeLabel p{background-color:#F5E6D8;}#mermaid-r1ae-r44 .edgeLabel rect{opacity:0.5;background-color:#F5E6D8;fill:#F5E6D8;}#mermaid-r1ae-r44 .labelBkg{background-color:rgba(245, 230, 216, 0.5);}#mermaid-r1ae-r44 .cluster rect{fill:#CC785C;stroke:hsl(15, 12.3364485981%, 48.0392156863%);stroke-width:1px;}#mermaid-r1ae-r44 .cluster text{fill:#3387a3;}#mermaid-r1ae-r44 .cluster span{color:#3387a3;}#mermaid-r1ae-r44 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"Anthropic Sans",system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:12px;background:#CC785C;border:1px solid hsl(15, 12.3364485981%, 48.0392156863%);border-radius:2px;pointer-events:none;z-index:100;}#mermaid-r1ae-r44 .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#191919;}#mermaid-r1ae-r44 rect.text{fill:none;stroke-width:0;}#mermaid-r1ae-r44 .icon-shape,#mermaid-r1ae-r44 .image-shape{background-color:#F5E6D8;text-align:center;}#mermaid-r1ae-r44 .icon-shape p,#mermaid-r1ae-r44 .image-shape p{background-color:#F5E6D8;padding:2px;}#mermaid-r1ae-r44 .icon-shape .label rect,#mermaid-r1ae-r44 .image-shape .label rect{opacity:0.5;background-color:#F5E6D8;fill:#F5E6D8;}#mermaid-r1ae-r44 .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#mermaid-r1ae-r44 .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#mermaid-r1ae-r44 .node .neo-node{stroke:#D9D8D5;}#mermaid-r1ae-r44 [data-look="neo"].node rect,#mermaid-r1ae-r44 [data-look="neo"].cluster rect,#mermaid-r1ae-r44 [data-look="neo"].node polygon{stroke:url(#mermaid-r1ae-r44-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#mermaid-r1ae-r44 [data-look="neo"].node path{stroke:url(#mermaid-r1ae-r44-gradient);stroke-width:1px;}#mermaid-r1ae-r44 [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#mermaid-r1ae-r44 [data-look="neo"].node .neo-line path{stroke:#D9D8D5;filter:none;}#mermaid-r1ae-r44 [data-look="neo"].node circle{stroke:url(#mermaid-r1ae-r44-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#mermaid-r1ae-r44 [data-look="neo"].node circle .state-start{fill:#000000;}#mermaid-r1ae-r44 [data-look="neo"].icon-shape .icon{fill:url(#mermaid-r1ae-r44-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#mermaid-r1ae-r44 [data-look="neo"].icon-shape .icon-neo path{stroke:url(#mermaid-r1ae-r44-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#mermaid-r1ae-r44 :root{--mermaid-font-family:"Anthropic Sans",system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}FastAPI docs~430 .md filesLoaderload_documentsChunkermarkdown-aware split~1000 char + overlapEmbedderBGE-small-en-v1.5384-dimQdrantcosine similarityDensetop-k via vector searchBM25 Indexbuilt at startupfrom Qdrant payloadsRetrieverselectorHybridRRF k=60Rerankerbge-reranker-basecross-encoderGeneratorgpt-4o-miniCited answer+ source chunks
Each stage lives in src/rag/ and is independently runnable (every module has a __main__ for testing in isolation):
StageFileWhat it doesLoadloader.pyReads markdown files, preserves title and section metadataChunkchunker.pyMarkdown-aware recursive split; ~1000-char chunks with 150-char overlap; prefixes each with "{title} > {header}" so headings carry into the embeddingEmbedembedder.pyBGE-small-en-v1.5 (384-dim, CPU-friendly, strong on technical content)Storestore.pyQdrant collection with cosine distance; stores chunk text and metadata in payloadRetrieve (dense)retriever.pyThin wrapper: query string → embedding → Qdrant top-kRetrieve (BM25)bm25_index.pyIn-process BM25 over chunk text; rebuilt at API startup from Qdrant payloads (single source of truth)Retrieve (hybrid)hybrid.pyCalls dense + BM25, fuses with Reciprocal Rank FusionRerankreranker.pyWraps a retriever; cross-encoder rescores the top-N candidatesGenerategenerator.pyBuilds the prompt with numbered context blocks; gpt-4o-mini emits the answer with [n] citationsServeapi.pyFastAPI app, lifespan-loaded retrievers, POST /ask and GET /healthUIui.pyStreamlit; talks to the API, shows answer + sources + latency
The retriever selector at the center of the diagram is the substantive bit. All three retrievers expose the same search(query: str, top_k: int, section: str | None) -> list[SearchResult] contract, so they're swappable from one env var (RETRIEVER=dense|hybrid|reranked) or one request parameter. The generator and API don't know or care which one is wired in.

Evaluation
The honest answer to "is hybrid better than dense?" or "does reranking help?" is: it depends. On what corpus, what queries, what eval set. So before any claims, the measurement.
The eval set
41 question/expected-chunk pairs in eval/dataset.json. Built in three passes:

LLM-drafted candidates. eval/generate_dataset.py pulls a random sample of 60 chunks from Qdrant, asks gpt-4o-mini to write one realistic developer question per chunk. ~50 candidates after the model skips reference-only chunks.
Curation. ~15% of LLM candidates turned out to be hallucinated mappings (a question about "real-time log streaming" paired with an editor-support doc) or unnatural meta-phrasings. Deleted those. Rephrased ~30% to use user vocabulary instead of doc vocabulary — the question "what's the best way to set up a current user dependency with OAuth2" became "how do I figure out who's logged in inside my endpoint."
Hand-written hard cases. Added 5 questions in pure user vocabulary that does not appear in the docs, to actually stress retrieval. Examples: "how do I let my frontend on a different domain talk to my API?" (the docs say "CORS"), "what's the safest way to store user passwords?" (the docs say "OAuth2 with hashing"), "how does FastAPI handle secure authentication?" (broad, multi-chunk).

After running the eval once, an artifact became obvious: gold-standard answers were over-strict. The reranker would put response-model.md::6 at rank 1 when the gold was ::7 — two sibling chunks in the same doc, both legitimate answers, but hit-rate marked it a miss. Went back and widened expected_chunk_ids to include obvious siblings for 6 questions, after reading each candidate chunk and judging would a developer who landed here find their answer. This is curation work that LLM bootstrapping cannot do, and it materially changes the numbers.
The tags field on each test case (identifier, conceptual, user-vocab, hard, easy) is for slicing results later — useful when investigating which retriever fails on which query type.
Metrics
Two per retriever:

hit@5 — for each query, did any expected chunk appear in the top-5? Boolean per query, averaged. Headline number, easy to interpret.
MRR — 1 / rank of the first expected chunk in the top-5, or 0 if not found. Averaged. Captures how high the right answer ranked — important for reranker eval since reranking is about precision-at-top.

Plus latency p50 and mean.
Results
41 questions, top-k=5:
Retrieverhit@5MRRp50 msmean msdense0.8050.6122638hybrid0.7810.5962424reranked0.7810.5527,9497,902
Three findings, each worth reading carefully.
Dense is the strongest retriever on this corpus. This was not what I expected. The textbook story — "dense throws away exact tokens, BM25 catches them, hybrid wins" — does not survive contact with BGE-small + FastAPI's docs. BGE-small is strong enough, and the docs coherent enough, that even rare identifiers like OAuth2PasswordBearer land in the right semantic neighborhood. BM25's lexical signal is real but redundant. On queries where the two retrievers agree (most of them), RRF is a no-op; on queries where BM25 elevates a lexically-similar-but-semantically-wrong chunk, hybrid degrades slightly. The net effect is a small MRR regression.
This isn't an indictment of hybrid retrieval. It's a calibration: hybrid's win shows up on identifier-heavy queries against a corpus with vocabulary the dense model handles poorly. Neither condition is strongly true here. The Block 5 work isn't wasted — it costs essentially nothing at runtime (24ms mean vs 38ms for dense) and provides cheap insurance — but its measured contribution on this eval is zero.
Reranking actively hurts. Same hit@5 as hybrid (0.781) but lower MRR (0.552 vs 0.596) at 200× the latency. Two contributors. First, the cross-encoder makes substantive but sometimes wrong judgments — on "how do I let my frontend on a different domain talk to my API?" it elevated a security chunk above the actual CORS chunks. Second, bge-reranker-base is trained primarily on MS MARCO web queries; it isn't tuned for docs-style retrieval and its notion of "relevance" doesn't always match what FastAPI's documentation is actually offering. Inspection of misses showed that on roughly 6 of 9 cases the reranker chose a sibling chunk to the gold answer (a measurement artifact, fixed by widening expected_chunk_ids); the remaining 3 were genuine errors.
The 7.9-second mean latency is dominated by 20 cross-encoder forward passes on CPU. Faster on GPU. Still not free.
The retrieval ceiling is set by query phrasing, not retriever choice. Several hand-written hard queries — "how do I let my frontend on a different domain talk to my API?", "how do I figure out who's logged in inside my endpoint?" — were failed by all three retrievers. The docs use "CORS" and "current user dependency," not "different domain" and "logged in." No amount of retriever sophistication closes that gap. The lever for these cases is query rewriting — preprocess the user's question with an LLM to bridge user vocabulary to doc vocabulary — not a better retriever.
What the numbers actually mean
Take all of this with several caveats:

41 questions is small. Differences below ~0.05 hit@5 are within noise; the dense/hybrid gap (0.024) is at that threshold.
One corpus, one model family. FastAPI's docs are unusually coherent and well-written for technical documentation. A noisier corpus (raw forum posts, code comments, multi-version manuals) would likely shift the picture toward BM25 and hybrid.
The eval set is balanced toward conceptual/user-vocab queries (28 of 41). Adding more pure-identifier queries — exact API names, error codes, decorators — would probably restore some hybrid advantage.
Off-the-shelf cross-encoder. A reranker fine-tuned on docs-style Q-chunk pairs would almost certainly beat the baseline; that's future work.

What this does support: dense as the default, hybrid as cheap insurance, reranking as an experiment that did not pay off on this specific setup. All three are kept in the codebase so the comparison is reproducible.
Reproducing
bash# Inside the running stack, with the indexer already run:
docker compose exec api uv run python eval/run_eval.py \
  --dataset eval/dataset.json --top-k 5

# Or skip the reranker for faster iteration:
docker compose exec api uv run python eval/run_eval.py \
  --dataset eval/dataset.json --top-k 5 --skip-reranker
Per-query results land in eval/results.csv. Sort by reciprocal_rank ascending and the failure modes become obvious.

Design decisions
The interesting choices, with the reasoning. Each one had a defensible alternative; this section is the why this one.
Embedding model: BGE-small-en-v1.5
384-dim, ~130MB, CPU-friendly. Strong on technical content (top of MTEB at this parameter count when I built this). The relevant comparison was BGE-base (768-dim, ~440MB) or OpenAI's text-embedding-3-small. Small was deliberate: the entire system runs on a laptop CPU in seconds, which makes iteration honest — the latency a recruiter sees on their machine is the latency I see on mine. Going bigger trades that for marginal MTEB gains the eval probably can't even resolve at n=41.
Vector store: Qdrant
Three options I weighed: Qdrant, Chroma, in-memory FAISS. Qdrant won for two reasons. First, it has native sparse-vector support, which would have made the hybrid retrieval implementation (Block 5) a server-side concern rather than an in-process one — a path worth knowing exists, even though I deliberately built BM25 in-process for visibility. Second, it ships as a Docker image with a sensible storage layout, which means the production deployment is one compose service, not "engineer your own persistence." Chroma is simpler at small scale but the persistence model is less clean. FAISS is the bare-metal option and has no payload storage, which would have forced a separate metadata store — unnecessary plumbing.
Hybrid fusion: Reciprocal Rank Fusion
The alternative was weighted-score fusion (score = α * cosine + (1-α) * bm25). Two problems with weighted score: cosine lives in ~[0,1] and BM25 is unbounded and query-dependent, so normalization is fiddly and corpus-specific; and the right α is unknowable without eval data per corpus. RRF sidesteps both by using only rank position: each chunk gets 1/(k + rank) from each list, summed. No score normalization, no per-corpus tuning. k=60 is the Cormack et al. default and the de-facto standard across Elasticsearch, Weaviate, and Qdrant. It also has the nice property that a chunk only one retriever finds gets a fair contribution from that one list, not zero — graceful degradation when the retrievers disagree.
BM25 implementation: in-process via bm25s
Qdrant supports server-side sparse vectors and would handle this natively. I deliberately built it in-process anyway, for two reasons. First, the mechanism stays visible — TF saturation, IDF weighting, length normalization — instead of becoming a library API call. Second, the corpus is small (1,359 chunks) and the BM25 index rebuilds at API startup in well under a second by scrolling Qdrant's payloads. The startup rebuild also enforces a single source of truth: whatever chunks exist in Qdrant get indexed by BM25, automatically, no possibility of drift.
The tokenizer is identifier-preserving: lowercase + runs of [a-z0-9_], no stemming, no stopwords. Default English tokenizers split OAuth2PasswordBearer into ["oauth", "password", "bearer"] — exactly destroying the rare token BM25 exists to catch. IDF handles common words automatically.
Reranker: retrieve-then-rerank, not full-corpus
Cross-encoders attend to query and chunk jointly — much more accurate per pair than bi-encoders, hundreds of times slower. Running the cross-encoder over all 1,359 chunks per query would take ~70 seconds; running it over the top 20 candidates from hybrid takes ~5–8s. The retrieve-then-rerank architecture pushes recall onto the cheap retrievers and reserves precision for the expensive one. This is the canonical setup and the right one.
The choice of bge-reranker-base over alternatives (ms-marco-MiniLM-L-6-v2, bge-reranker-large) was a guess: same family as the embedder, mid-sized, runs on CPU. The eval suggests it was the wrong guess for this corpus. The architecture stays; the model is replaceable in one line.
Why no framework
LangChain, LlamaIndex, and Haystack all do RAG. None of them appear in this codebase. The reason isn't ideology — it's that frameworks hide exactly the decisions this project exists to make visible. The chunking strategy, the embedding step, the score fusion, the rerank logic — those are the engineering. Treating them as black-box library calls means never having to explain why a chunk has a 150-char overlap, or what RRF actually computes, or why gpt-4o-mini's prompt is shaped the way it is. The cost of going framework-free is ~500 lines of source code that someone else could have written. The benefit is being able to defend every line.
The system uses real libraries (qdrant-client, sentence-transformers, bm25s, openai) — what it avoids is frameworks that compose them for you.

Project structure
docs-rag/
├── src/rag/
│   ├── loader.py          # Stage 1: walk the docs folder, return Document objects
│   ├── chunker.py         # Stage 2: markdown-aware split, header prefixes, overlap
│   ├── embedder.py        # Stage 3: BGE-small-en-v1.5 wrapper
│   ├── store.py           # Stage 4: Qdrant collection (upsert, search, count)
│   ├── retriever.py       # DenseRetriever — VectorStore + Embedder, string in / chunks out
│   ├── bm25_index.py      # BM25Index — in-process sparse retrieval over Qdrant payloads
│   ├── hybrid.py          # HybridRetriever — dense + BM25 fused via RRF
│   ├── reranker.py        # CrossEncoderReranker + RerankingRetriever — wraps any retriever
│   ├── generator.py       # gpt-4o-mini with grounded prompt + [n] citation rules
│   ├── api.py             # FastAPI: lifespan-loaded retrievers, /ask, /health
│   └── ui.py              # Streamlit chat UI
│
├── eval/
│   ├── dataset.json       # 41 curated test cases — single source of truth for measurement
│   ├── generate_dataset.py  # LLM-bootstraps candidates; outputs JSONL for curation
│   ├── inspect_chunks.py  # Dump chunk text by chunk_id — used during sibling-chunk curation
│   └── run_eval.py        # Runs hit@k and MRR across all retrievers, writes results.csv
│
├── docker/
│   ├── Dockerfile         # Single image, three roles via entrypoint
│   └── entrypoint.sh      # Dispatches: api | ui | indexer
│
├── data/raw/fastapi/      # The corpus — FastAPI's docs/ tree, baked into the image
├── qdrant_storage/        # Qdrant's persistent data (bind-mounted, not in git)
├── docker-compose.yml     # Three services + indexer one-shot, all bound to 127.0.0.1
├── pyproject.toml         # uv-managed deps
└── README.md              # This file
Every module under src/rag/ has a __main__ block that runs it in isolation against the real corpus — useful when iterating on one stage without spinning up the full stack. uv run python src/rag/chunker.py reports chunk statistics; uv run python src/rag/bm25_index.py runs sample queries through BM25 only; and so on.

Configuration
Environment variables
env# Required
OPENAI_API_KEY=sk-...           # for the generator

# Optional (containerized defaults work for compose)
QDRANT_URL=http://localhost:6333  # falls back to localhost outside Docker
OPENAI_MODEL=gpt-4o-mini          # any chat-completion model

# Retriever selection
RETRIEVER=dense                   # dense | hybrid | reranked  (default: dense)
RETRIEVER sets the server-wide default. Any POST /ask request can override per-call with "retriever": "hybrid" in the body — useful for live A/B comparison and for the eval harness.
Local development without Docker
The entire pipeline runs natively if you'd rather skip Docker:
bashuv sync
docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
uv run python -m rag.store              # indexes the corpus
uv run uvicorn rag.api:app --reload     # serves the API
uv run streamlit run src/rag/ui.py      # serves the UI
The only thing Docker provides that the host doesn't is the pre-downloaded models — natively, BGE and the reranker pull from HuggingFace on first use.

Roadmap
Honest list, in roughly the order they'd actually move the system:

Query rewriting. The single biggest gap from the eval — several queries failed across all three retrievers because the user's vocabulary doesn't appear in the docs. Preprocessing the question with gpt-4o-mini ("rewrite this query in terms a FastAPI doc would use") would directly attack that ceiling. This is the most technically interesting block left and probably the highest-impact change.
Identifier-heavy eval set. Currently 28 of 41 questions are conceptual/user-vocab; only 3 are pure-identifier queries. Adding 20–30 questions like Depends, BackgroundTasks, HTTPException, response_model_exclude_unset would actually stress BM25's strength and tell us whether hybrid is truly zero-value here or just unmeasured.
Answer-quality eval. Retrieval eval (hit@k, MRR) only tells you whether the right chunks were retrieved. It says nothing about whether the LLM faithfully used them. The standard fix is LLM-as-judge — have a strong model score each answer for faithfulness and answer-relevance against the retrieved context. Worth doing if I want to claim anything about end-to-end quality, not just retrieval quality.
Reranker swap or fine-tune. bge-reranker-base underperformed; the architecture is right, the model probably isn't. Two paths: try ms-marco-MiniLM-L-6-v2 (smaller, faster, different training mix), or fine-tune any cross-encoder on synthetic Q-chunk pairs drawn from the corpus. The latter is the right answer for a real production system; for this project, the swap is the cheaper experiment.
Multi-corpus support. Right now the system is hard-coded to one FastAPI snapshot. Real docs-RAG products handle many products and versions in one instance, with a tenant/version filter on retrieval. The chunking and indexing pipeline don't change; the API and Qdrant filtering do.
Streaming responses. The generator returns a complete AskResponse — fine for a 4-second latency, but for reranked at 8s+ it feels broken. Switching to streaming would mask latency and is a one-screen change in api.py and ui.py.
Observability. Request logs go to stdout; there's no histogram, no /stats endpoint, no distinction between retrieval latency and LLM latency. Real production needs this; portfolio version arguably doesn't.

Known limitations

Cross-encoder latency on CPU is harsh. 8 seconds is hostile for interactive use. Acceptable for an experiment; not for a product. GPU or a smaller reranker would fix it.
The corpus is a frozen snapshot. No incremental indexing — docker compose run --rm indexer rebuilds from scratch. Fine for ~1,400 chunks; the wrong design at 100,000.
No auth on the API. The whole point of binding to 127.0.0.1 is that there's no auth layer; any user on the host has full access. Internet exposure would require at minimum an API key middleware.
The eval set is mine, alone. No inter-annotator agreement, no second opinion. Some "expected" chunk choices reflect my judgment of what answers a question, and a different reader might pick differently. The metric is a measurement, not ground truth.


Acknowledgments

The FastAPI documentation (Sebastián Ramírez and contributors) is the corpus. It's also the source of the framework that runs the API — pleasingly recursive.
BAAI for the BGE embedding and reranker models. Strong, small, free.
Qdrant for the vector database. Sensible defaults, clean Docker story, supports sparse + dense.
bm25s (Xing Han Lu) for a fast modern BM25 in Python. Saved me writing 80 lines of TF/IDF.
The Cormack et al. paper on Reciprocal Rank Fusion (2009), which is one of those rare publications where the idea is simple, the implementation is 10 lines, and it's still the default 16 years later.