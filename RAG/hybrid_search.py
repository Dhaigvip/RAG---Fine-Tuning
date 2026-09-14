"""
hybrid_search.py — combine BM25 keyword search + FAISS vector search via
Reciprocal Rank Fusion, rerank the fused candidates, then PROMOTE surviving
matches from their small child chunk up to their larger parent chunk before
returning them. Two reranker BACKENDS are implemented side by side — see
reranking-strategies.md for the full writeup of why both exist:

  - "local"   (default): sentence-transformers CrossEncoder, runs on-device.
  - "bedrock": Amazon Bedrock Rerank API (Cohere/Amazon rerank models) — the
    original plan, currently blocked for this AWS account by an
    Organizations Service Control Policy (SCP). Left in place, unused by
    default, so switching back is a one-line env var change once model
    access is granted — no code to resurrect.

Pick the backend with TOOL_RAG_RERANK_BACKEND=local|bedrock in .env.

Mechanism, in order (see retrieval-concepts.md, reranking-strategies.md, and
parent-child-retrieval.md for the full explanations):
1. Load CHILD chunks + the FAISS index already built by faiss_search.py
   --build (children are the matching unit — see parent-child-retrieval.md).
   Also load PARENTS (chunks.jsonl) as a plain, unembedded lookup list.
2. BM25: score every child against the query by term matching — catches
   exact terms (names, commands, acronyms) that embeddings can blur.
3. Vector search: same FAISS cosine-similarity search as faiss_search.py —
   catches paraphrases/synonyms that BM25 misses entirely (calls Bedrock
   for the query embedding regardless of which rerank backend is active —
   embeddings were never affected by the SCP block, only rerank models).
   As of Sept 14, what actually gets embedded is the query AFTER
   query_transform.py's transform_query_for_embedding() — by default that's
   a HyDE-generated hypothetical answer, not the raw query text, aimed
   directly at the paraphrase-vs-keyword score gap this project already
   measured (+4.90 vs +0.53 for the same correct chunk — see
   reranking-strategies.md and query-transformation-strategies.md).
4. Reciprocal Rank Fusion (RRF): merge the two ranked lists using RANKS,
   not raw scores — BM25 scores and cosine similarities live on
   incomparable scales, but ranks are always comparable.
5. Rerank: send the fused candidates to whichever backend is selected —
   it looks at the query and each candidate together (unlike the bi-encoder
   similarity used for vector search), producing a more precise final
   ranking than either retrieval signal alone. A per-backend score threshold
   (LOCAL_MIN_SCORE / BEDROCK_MIN_SCORE) is then applied so a query with no
   genuinely relevant chunk returns fewer results instead of being padded
   with confidently-irrelevant ones — see reranking-strategies.md. NOTE:
   this now reranks the FULL fused candidate pool, not just FINAL_TOP_K —
   see promote_to_parents() below for why.
6. Promote: map each surviving CHILD back to its PARENT via parent_id,
   dedup (multiple children can share one parent — keep only the
   highest-scoring one), then slice to FINAL_TOP_K unique parents. The
   parent's full text — not the small child fragment that matched — is
   what actually gets returned as context. This is the "small-to-big" swap
   parent-child-retrieval.md describes: small unit for matching, large unit
   for what's shown.

Requires: pip install rank-bm25 sentence-transformers
Run `python embed.py` then `python faiss_search.py --build` first if you
haven't already (embeds/indexes children by default — see embed.py).

Usage:
    python hybrid_search.py <query text>
"""

import json
import os
import sys
from pathlib import Path

import boto3
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import query_cache
from embed import embed_text, REGION, PROFILE, MODEL_ID as EMBED_MODEL_ID
from faiss_search import load_index
from query_transform import transform_query_for_embedding, QUERY_TRANSFORM, HYDE_MODEL_ID

RRF_K = 60            # standard RRF constant
CANDIDATE_POOL = 10   # fused candidates sent to the reranker
FINAL_TOP_K = 3        # unique PARENTS returned after promotion (not children)

PARENTS_PATH = Path(__file__).parent / "output" / "chunks.jsonl"

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


def load_parents() -> list:
    """Load chunks.jsonl — the PARENT chunks, as plain lookup records.

    Added Sept 14 (parent/child retrieval stage 2). Parents are never
    embedded or indexed (see parent-child-retrieval.md — only children are
    the matching unit), so this is just a flat JSON-lines read, no FAISS/
    embedding involved. A child's parent_id is its position in THIS list —
    chunking.py guarantees that ordering matches embed.py/faiss_search.py's
    processing order for chunks.jsonl, so parents[parent_id] is a direct,
    O(1) lookup with no separate ID-matching logic needed."""
    lines = PARENTS_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def bm25_rank(query: str, metadata: list) -> list:
    """Return chunk indices ranked best-to-worst by BM25 score.

    BM25 is pure text matching — it never sees an embedding vector. `metadata`
    here (loaded from faiss_metadata.jsonl) is a slightly misleading name: it
    holds each chunk's actual text plus true metadata (source_file,
    header_path, and — as of Sept 14 — parent_id/child_index for children)
    bundled together — everything about the chunk EXCEPT its embedding, which
    lives only inside the FAISS index file. This function reads c["text"]
    (the real content) and tokenizes it into words; it has no access to, and
    no use for, the vector representation at all. As of Sept 14 this operates
    on CHILD chunks (the matching unit), not parents."""
    tokenized_corpus = [c["text"].lower().split() for c in metadata]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    return list(np.argsort(scores)[::-1])


def vector_rank(query: str, index, k: int, cache: dict = None) -> list:
    """Return chunk indices ranked best-to-worst by FAISS vector similarity.

    The exact mirror image of bm25_rank: this function embeds the query
    (embed_text) and compares that vector against every vector stored in the
    FAISS index via cosine similarity. It never looks at any chunk's raw text
    — from here, a chunk is purely a point in 1024-dimensional space. As of
    Sept 14 the index holds CHILD vectors (see faiss_search.py).

    bm25_rank and vector_rank are deliberately two independent signals
    computed from two different representations of the same chunk (its words
    vs. its embedding) — that's why fusing them below catches more than
    either alone: they're blind to different things, not redundant.

    Added Sept 14 (query transformation): what actually gets embedded here
    is transform_query_for_embedding(query), not the raw query directly —
    by default that's HyDE (a generated hypothetical answer, phrased like the
    corpus is phrased, embeds closer to the real match than a terse question
    does), with a built-in fallback to the raw query on any failure. BM25
    above deliberately does NOT go through this — see
    docs/query-transformation-strategies.md.

    Added Sept 14 (evaluation harness): optional `cache` dict (see
    query_cache.py) — when given, skips BOTH the HyDE generation call and
    the embedding call entirely on a hit, instead of re-triggering two real
    Bedrock calls (one of them non-deterministic) for a query that's been
    embedded before under the exact same transform mode and models.
    cache=None (the default) preserves the exact existing behavior for
    interactive/CLI use — always embeds fresh, so HyDE's natural variation
    is preserved there; caching is opt-in, used by evaluate_retrieval.py
    specifically because it re-runs the same fixed question set repeatedly.
    See docs/evaluation-strategies.md for the full reasoning."""
    cache_key = None
    if cache is not None:
        cache_key = query_cache.cache_key(query, QUERY_TRANSFORM, EMBED_MODEL_ID, HYDE_MODEL_ID)
        cached_vector = query_cache.get_cached_vector(cache, cache_key)
        if cached_vector is not None:
            query_vector = np.array([cached_vector], dtype="float32")
            _, indices = index.search(query_vector, k)
            return [int(i) for i in indices[0] if i != -1]

    embedding_input = transform_query_for_embedding(query)
    query_vector, _ = embed_text(embedding_input)

    if cache is not None:
        query_cache.set_cached_vector(cache, cache_key, query_vector)

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
    top_k when a threshold is active and the corpus has no better matches.
    NOTE: model.predict() below already scores every candidate in
    candidate_indices regardless of top_k — so calling this with a larger
    top_k (as search() now does, to support promotion — see
    promote_to_parents()) costs nothing extra; only the final slice changes."""
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
    from. Bedrock is asked for numberOfResults=top_k directly, so — unlike
    rerank_local — a larger top_k here DOES mean Bedrock itself returns more
    results (it doesn't score every candidate up front the way the local
    cross-encoder loop does); still cheap at this candidate-pool size.

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


def promote_to_parents(reranked_children: list, child_metadata: list, top_k: int) -> list:
    """Map surviving CHILD matches back to their PARENT, dedup, slice to top_k.

    Added Sept 14 (parent/child retrieval stage 2) — this is the actual
    "small-to-big" swap: matching happened on small children, but this
    promotes each match to the id of its larger parent before anything gets
    shown or handed to generation. See parent-child-retrieval.md for the
    full design.

    Args:
        reranked_children: (child_idx, score) pairs, already sorted best-first
            (i.e. the output of rerank(), called with a top_k LARGE ENOUGH to
            cover the whole fused candidate pool — see search() below for why
            this must not already be sliced to FINAL_TOP_K).
        child_metadata: the child records list (from faiss_metadata.jsonl),
            positionally aligned with reranked_children's indices — each
            record must carry "parent_id".
        top_k: how many UNIQUE parents to return.

    Returns:
        (parent_id, score) pairs, best first, deduped, sliced to top_k. The
        score carried through is the matching child's rerank score — since
        the input is already sorted best-first, the FIRST child seen for a
        given parent_id is necessarily that parent's best-scoring child, so
        "keep first occurrence, skip repeats" is exactly equivalent to "keep
        the max score per parent" — just simpler to write.

    Pure function (plain data in, plain data out, no AWS/model/FAISS
    dependency) — see tests/test_hybrid_search.py."""
    seen_parent_ids = set()
    promoted = []
    for child_idx, score in reranked_children:
        parent_id = child_metadata[child_idx]["parent_id"]
        if parent_id in seen_parent_ids:
            continue  # a higher-or-equal-scoring sibling child already claimed this parent
        seen_parent_ids.add(parent_id)
        promoted.append((parent_id, score))
        if len(promoted) >= top_k:
            break
    return promoted


def search(query: str, verbose: bool = True, cache: dict = None):
    """Run the full retrieval pipeline for one query.

    Added Sept 14 (evaluation harness): `verbose` (default True, so every
    existing CLI/manual-testing call to search() behaves exactly as before)
    lets evaluate_retrieval.py run this in a loop over a whole eval set
    without a wall of BM25/vector/RRF debug printing per question — see
    docs/evaluation-strategies.md. Also added: this now RETURNS the promoted
    (parent_id, score) list (empty list when nothing passed the relevance
    threshold) instead of only printing it, so a caller can actually check
    the result programmatically rather than eyeballing stdout — the eval
    harness is exactly that caller.

    Also added: `cache` (default None, unchanged behavior) is passed straight
    through to vector_rank() — see its docstring and query_cache.py for why
    this exists (skip redundant HyDE + embedding Bedrock calls on repeated
    eval runs) and why it's opt-in rather than the default."""
    index, child_metadata = load_index()
    parents = load_parents()

    bm25_ranked = bm25_rank(query, child_metadata)
    vector_ranked = vector_rank(query, index, k=len(child_metadata), cache=cache)
    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked])
    candidate_indices = [idx for idx, _ in fused[:CANDIDATE_POOL]]

    if verbose:
        print(f"\nQuery: {query!r}")
        print(f"BM25 top:    {[child_metadata[i]['source_file'] for i in bm25_ranked[:3]]}")
        print(f"Vector top:  {[child_metadata[i]['source_file'] for i in vector_ranked[:3]]}")
        print(f"RRF-fused candidates -> reranker ({RERANK_BACKEND}): "
              f"{[child_metadata[i]['source_file'] for i in candidate_indices]}")

    # Rerank the FULL candidate pool (not just FINAL_TOP_K) — promotion below
    # can collapse multiple children onto one parent via dedup, so slicing to
    # FINAL_TOP_K before promoting could silently under-fill the final result
    # even when a distinct-parent match was sitting just below the cutoff.
    reranked_children = rerank(query, candidate_indices, child_metadata, top_k=len(candidate_indices))

    if not reranked_children:
        if verbose:
            print(f"\nNo candidates passed the relevance threshold — nothing confidently "
                  f"relevant found for this query in the current corpus.\n")
        return []

    promoted = promote_to_parents(reranked_children, child_metadata, top_k=FINAL_TOP_K)

    if verbose:
        print(f"\n{len(reranked_children)} child match(es) passed the relevance threshold "
              f"-> {len(promoted)} unique parent(s) after promotion (requested {FINAL_TOP_K}):\n")
        for rank, (parent_id, score) in enumerate(promoted, start=1):
            parent = parents[parent_id]
            # Show which child actually matched, for transparency/debugging — the
            # first reranked child whose parent_id equals this parent's is exactly
            # the one promote_to_parents() picked (same "first occurrence wins" rule).
            matched_child_idx = next(i for i, _ in reranked_children if child_metadata[i]["parent_id"] == parent_id)
            child_preview = child_metadata[matched_child_idx]["text"].replace("\n", " ")[:100]
            print(f"{rank}. relevance={score:.4f}  [{parent['source_file']}] {parent['header_path'] or '(no header)'}")
            print(f"   matched via child: {child_preview}...")
            # ...but what's actually returned as context is the PARENT's full text.
            parent_preview = parent["text"].replace("\n", " ")[:200]
            print(f"   parent context:    {parent_preview}...\n")

    return promoted


if __name__ == "__main__":
    if len(sys.argv) > 1:
        search(" ".join(sys.argv[1:]))
    else:
        print("Usage: python hybrid_search.py <query text>")
