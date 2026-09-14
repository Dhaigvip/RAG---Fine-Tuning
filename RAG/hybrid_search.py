"""
hybrid_search.py — combine BM25 keyword search + FAISS vector search via
Reciprocal Rank Fusion, then rerank the fused candidates. Two reranker
BACKENDS are implemented side by side — see reranking-strategies.md for the
full writeup of why both exist:

  - "local"   (default): sentence-transformers CrossEncoder, runs on-device.
  - "bedrock": Amazon Bedrock Rerank API (Cohere/Amazon rerank models) — the
    original plan, currently blocked for this AWS account by an
    Organizations Service Control Policy (SCP). Left in place, unused by
    default, so switching back is a one-line env var change once model
    access is granted — no code to resurrect.

Pick the backend with TOOL_RAG_RERANK_BACKEND=local|bedrock in .env.

Mechanism, in order (see retrieval-concepts.md and reranking-strategies.md
for the full explanations):
1. Load chunks + the FAISS index already built by faiss_search.py --build.
2. BM25: score every chunk against the query by term matching — catches
   exact terms (names, commands, acronyms) that embeddings can blur.
3. Vector search: same FAISS cosine-similarity search as faiss_search.py —
   catches paraphrases/synonyms that BM25 misses entirely (calls Bedrock
   for the query embedding regardless of which rerank backend is active —
   embeddings were never affected by the SCP block, only rerank models).
4. Reciprocal Rank Fusion (RRF): merge the two ranked lists using RANKS,
   not raw scores — BM25 scores and cosine similarities live on
   incomparable scales, but ranks are always comparable.
5. Rerank: send the fused top candidates to whichever backend is selected —
   it looks at the query and each candidate together (unlike the bi-encoder
   similarity used for vector search), producing a more precise final
   ranking than either retrieval signal alone. A per-backend score threshold
   (LOCAL_MIN_SCORE / BEDROCK_MIN_SCORE) is then applied so a query with no
   genuinely relevant chunk returns fewer than FINAL_TOP_K results instead of
   being padded with confidently-irrelevant ones — see reranking-strategies.md.

Requires: pip install rank-bm25 sentence-transformers
Run `python faiss_search.py --build` first if you haven't already.

Usage:
    python hybrid_search.py <query text>
"""

import os
import sys

import boto3
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from embed import embed_text, REGION, PROFILE
from faiss_search import load_index

RRF_K = 60            # standard RRF constant
CANDIDATE_POOL = 10   # fused candidates sent to the reranker
FINAL_TOP_K = 3

# Which reranker implementation to use — "local" needs no AWS access at all
# and is the default since the "bedrock" backend is currently SCP-blocked on
# this account. Flip to "bedrock" in .env once rerank model access opens up.
RERANK_BACKEND = os.environ.get("TOOL_RAG_RERANK_BACKEND", "local")

# Local cross-encoder model — configurable via .env. ms-marco-MiniLM-L-6-v2
# is small and fast (~80MB download); BAAI/bge-reranker-base is stronger but
# larger, worth trying if quality matters more than speed at this scale.
RERANK_MODEL_ID = os.environ.get("TOOL_RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Bedrock rerank model — unused while RERANK_BACKEND=local, kept configurable
# for when the backend is flipped back. cohere.rerank-v3-5:0 and
# amazon.rerank-v1:0 are both confirmed available in eu-central-1.
BEDROCK_RERANK_MODEL_ID = os.environ.get("TOOL_RAG_BEDROCK_RERANK_MODEL", "cohere.rerank-v3-5:0")

# Relevance-score cutoff applied AFTER ranking, BEFORE slicing to top_k — see
# reranking-strategies.md ("top_k: fixed count vs relevance threshold") for the
# full writeup. Two live test queries against this test corpus both showed the
# same pattern: only one chunk was ever genuinely relevant, but fixed top-k
# still padded the result with two confidently-irrelevant chunks (scores below
# -9). A threshold fixes that, but the two backends' scores are NOT on the same
# scale, so each gets its OWN cutoff, independently configurable:
#   - local (MiniLM cross-encoder): unbounded raw logits. Both test runs put
#     the one correct match clearly positive (+4.90, +0.53) and every wrong
#     match clearly negative (-9.7 to -11.3) — 0.0 cleanly separates them on
#     the evidence gathered so far. This is a MODEL-SPECIFIC empirical
#     observation from 2 queries on 6 chunks, not a general law of MS MARCO
#     cross-encoders — revisit if a bigger corpus/more queries disagree.
#   - bedrock: normalized 0-1 relevance scores, never exercised yet (SCP-
#     blocked) — left unset (None = no filtering) rather than guessing a
#     number with zero evidence behind it. Set TOOL_RAG_RERANK_MIN_SCORE_BEDROCK
#     once this backend is actually usable and has been observed on real
#     queries.
# Either can be disabled by leaving it unset in .env / passing an empty string
# — None means "keep old fixed-top-k behavior," never filtering.
_local_min = os.environ.get("TOOL_RAG_RERANK_MIN_SCORE_LOCAL", "0.0")
LOCAL_MIN_SCORE = float(_local_min) if _local_min else None
_bedrock_min = os.environ.get("TOOL_RAG_RERANK_MIN_SCORE_BEDROCK", "")
BEDROCK_MIN_SCORE = float(_bedrock_min) if _bedrock_min else None

_cross_encoder = None  # lazy-loaded so just importing this module doesn't trigger a model download
_bedrock_agent = None  # lazy-loaded so a "local"-backend run never needs a bedrock-agent-runtime client


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"Loading local cross-encoder '{RERANK_MODEL_ID}' (first run downloads it)...")
        _cross_encoder = CrossEncoder(RERANK_MODEL_ID)
    return _cross_encoder


def _get_bedrock_agent():
    global _bedrock_agent
    if _bedrock_agent is None:
        session = boto3.Session(profile_name=PROFILE) if PROFILE else boto3.Session()
        _bedrock_agent = session.client("bedrock-agent-runtime", region_name=REGION)
    return _bedrock_agent


def bm25_rank(query: str, metadata: list) -> list:
    """Return chunk indices ranked best-to-worst by BM25 score.

    BM25 is pure text matching — it never sees an embedding vector. `metadata`
    here (loaded from faiss_metadata.jsonl) is a slightly misleading name: it
    holds each chunk's actual text plus true metadata (source_file,
    header_path, chunk_index) bundled together — everything about the chunk
    EXCEPT its embedding, which lives only inside the FAISS index file. This
    function reads c["text"] (the real content) and tokenizes it into words;
    it has no access to, and no use for, the vector representation at all."""
    tokenized_corpus = [c["text"].lower().split() for c in metadata]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    return list(np.argsort(scores)[::-1])


def vector_rank(query: str, index, k: int) -> list:
    """Return chunk indices ranked best-to-worst by FAISS vector similarity.

    The exact mirror image of bm25_rank: this function embeds the query
    (embed_text) and compares that vector against every vector stored in the
    FAISS index via cosine similarity. It never looks at any chunk's raw text
    — from here, a chunk is purely a point in 1024-dimensional space.

    bm25_rank and vector_rank are deliberately two independent signals
    computed from two different representations of the same chunk (its words
    vs. its embedding) — that's why fusing them below catches more than
    either alone: they're blind to different things, not redundant."""
    query_vector, _ = embed_text(query)
    query_vector = np.array([query_vector], dtype="float32")
    _, indices = index.search(query_vector, k)
    return [int(i) for i in indices[0] if i != -1]


def apply_min_score(ranked: list, min_score, top_k: int) -> list:
    """Given (index, score) pairs already sorted best-first, drop anything
    below min_score, then slice to top_k. min_score=None disables filtering
    entirely (the old always-return-exactly-top_k behavior) — this is what
    makes the fixed-top-k vs relevance-threshold choice a one-value toggle
    instead of two code paths. Pure function (no AWS, no model, no I/O), so
    it's cheap to unit test directly — see tests/test_hybrid_search.py."""
    if min_score is not None:
        ranked = [pair for pair in ranked if pair[1] >= min_score]
    return ranked[:top_k]


def reciprocal_rank_fusion(rank_lists: list, k: int = RRF_K) -> list:
    """Merge multiple ranked lists of chunk indices into one fused ranking."""
    scores = {}
    for ranked in rank_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def rerank_local(query: str, candidate_indices: list, metadata: list, top_k: int) -> list:
    """Rerank candidates locally with a cross-encoder (sentence-transformers) —
    no AWS call, no SCP restrictions, entirely offline after the model's first
    download. A cross-encoder takes (query, document) PAIRS as input — unlike
    the bi-encoder used for vector search, which embeds each side separately,
    this model reads both texts together and outputs one relevance score per
    pair directly. That's why it's more precise but too slow to run over an
    entire corpus, and why it only sees the fused candidate pool, not everything.

    Applies LOCAL_MIN_SCORE after ranking — see apply_min_score() and the
    "top_k: fixed count vs relevance threshold" note in reranking-strategies.md.

    Returns (index, relevance_score) tuples, best first — may be FEWER than
    top_k when a threshold is active and the corpus has no better matches."""
    model = _get_cross_encoder()
    documents = [metadata[i]["text"] for i in candidate_indices]
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)

    ranked = sorted(zip(candidate_indices, scores), key=lambda pair: pair[1], reverse=True)
    return apply_min_score(ranked, LOCAL_MIN_SCORE, top_k)


def rerank_bedrock(query: str, candidate_indices: list, metadata: list, top_k: int) -> list:
    """Rerank candidates via the Bedrock Rerank API (Cohere/Amazon rerank
    models) — same cross-encoder idea as rerank_local, but hosted on AWS
    instead of running on this machine. Currently blocked on this account by
    an Organizations SCP (see reranking-strategies.md) — kept here, unused
    by default, so it's a one-line RERANK_BACKEND flip to restore once model
    access is granted, instead of rewriting this from scratch.

    Applies BEDROCK_MIN_SCORE after ranking (see apply_min_score()) — unset by
    default since this backend has zero real usage yet to calibrate a cutoff
    from. Bedrock is asked for numberOfResults=top_k directly, so this filter
    can only ever shrink what comes back, never reorder or grow it.

    Returns (index, relevance_score) tuples, best first."""
    bedrock_agent = _get_bedrock_agent()
    documents = [metadata[i]["text"] for i in candidate_indices]

    response = bedrock_agent.rerank(
        queries=[{"type": "TEXT", "textQuery": {"text": query}}],
        sources=[
            {"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": doc}}}
            for doc in documents
        ],
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {
                    "modelArn": f"arn:aws:bedrock:{REGION}::foundation-model/{BEDROCK_RERANK_MODEL_ID}"
                },
                "numberOfResults": min(top_k, len(documents)),
            },
        },
    )

    ranked = [(candidate_indices[r["index"]], r["relevanceScore"]) for r in response["results"]]
    return apply_min_score(ranked, BEDROCK_MIN_SCORE, top_k)


def rerank(query: str, candidate_indices: list, metadata: list, top_k: int) -> list:
    """Dispatch to whichever backend RERANK_BACKEND selects. Both functions
    share the same signature and return shape by design, so callers (search()
    below) never need to know or care which one actually ran."""
    if RERANK_BACKEND == "bedrock":
        return rerank_bedrock(query, candidate_indices, metadata, top_k)
    return rerank_local(query, candidate_indices, metadata, top_k)


def search(query: str):
    index, metadata = load_index()

    bm25_ranked = bm25_rank(query, metadata)
    vector_ranked = vector_rank(query, index, k=len(metadata))
    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked])
    candidate_indices = [idx for idx, _ in fused[:CANDIDATE_POOL]]

    print(f"\nQuery: {query!r}")
    print(f"BM25 top:    {[metadata[i]['source_file'] for i in bm25_ranked[:3]]}")
    print(f"Vector top:  {[metadata[i]['source_file'] for i in vector_ranked[:3]]}")
    print(f"RRF-fused candidates -> reranker ({RERANK_BACKEND}): {[metadata[i]['source_file'] for i in candidate_indices]}")

    reranked = rerank(query, candidate_indices, metadata, top_k=FINAL_TOP_K)

    if not reranked:
        print(f"\nNo candidates passed the relevance threshold — nothing confidently "
              f"relevant found for this query in the current corpus.\n")
        return

    print(f"\nFinal reranked top {len(reranked)} (requested {FINAL_TOP_K}):\n")
    for rank, (idx, score) in enumerate(reranked, start=1):
        chunk = metadata[idx]
        print(f"{rank}. relevance={score:.4f}  [{chunk['source_file']}] {chunk['header_path'] or '(no header)'}")
        preview = chunk["text"].replace("\n", " ")[:150]
        print(f"   {preview}...\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        search(" ".join(sys.argv[1:]))
    else:
        print("Usage: python hybrid_search.py <query text>")
