"""
evaluate_retrieval.py — labeled-eval-set harness for the RETRIEVAL half of
this pipeline (BM25 + vector search + RRF fusion + rerank + parent/child
promotion). Scores hybrid_search.py's search() against known-correct
answers instead of eyeballing one or two manual test queries.

Problem statement: every retrieval-affecting change made so far (chunk
size, CANDIDATE_POOL, the rerank relevance threshold, parent/child
promotion, HyDE) was judged by running one or two manual queries and
reading the printed output. That's genuinely how real bugs got caught
(the promotion-ordering bug, the relevance-threshold gap) — but it doesn't
scale and it can't catch a regression in a query nobody happened to type
that day. See docs/evaluation-strategies.md for the full landscape and why
this is retrieval-only for now: faithfulness/answer-relevancy (RAGAS's
generation-quality metrics) need an actual generated ANSWER to check for
grounding, and this project has no generation step yet (Sept 15-16 on the
plan) — there's nothing to compute those against yet.

Solution: a hand-labeled JSON eval set (eval/retrieval_eval_set.json) of
(query, expected_source_file[, expected_header_contains]) pairs built from
the real content of the 5 test notes, run through search(), scored with
two metrics that need no LLM judge at all:

  - Hit Rate@k: fraction of questions where the expected source appears
    ANYWHERE in the FINAL_TOP_K results returned. Simple, matches what a
    user actually sees ("was the right thing in there at all").
  - MRR (Mean Reciprocal Rank): mean of 1/rank of the expected source
    across all questions (0 if it never appears). More sensitive than Hit
    Rate to POSITION — rank 1 vs rank 3 look identical to Hit Rate but are
    very different in practice (rank 3 means 2 wrong/irrelevant chunks
    would reach generation ahead of the right one).

This is the same idea as RAGAS's non-LLM context precision/recall (compare
retrieved results against a known-correct reference by identity) — same
methodology, hand-implemented here to avoid pulling in langchain/
langchain-aws just to wrap a judge LLM this project doesn't even need for
these particular metrics. See docs/evaluation-strategies.md for the full
framework comparison (RAGAS/DeepEval/TruLens) and why each was deferred.

Query embedding cache (added Sept 14): this harness re-runs the SAME fixed
question set after every retrieval-affecting pipeline change — and none of
those changes (CANDIDATE_POOL, the reranker, LOCAL_MIN_SCORE, chunk size)
touch query embedding at all. Without caching, every re-run would burn a
real HyDE generation call AND a real Titan embedding call per question for
work whose result can't have changed — see query_cache.py's docstring for
the full reasoning. This is the one place in the project that opts into
that cache; the interactive CLI (`python hybrid_search.py <query>`) stays
uncached on purpose, so ad hoc querying still gets HyDE's real variation.

Run: python evaluate_retrieval.py   (from RAG/)
Re-run this after any change to CANDIDATE_POOL, FINAL_TOP_K, the rerank
model/backend, LOCAL_MIN_SCORE, chunk size, or query transformation — that
comparison across runs is the whole point; see docs/evaluation-strategies.md.
Delete eval/query_embedding_cache.json by hand to force a full re-embed
(e.g. after changing HyDE's prompt template, which isn't part of the cache
key — see query_cache.py).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from hybrid_search import search, load_parents
from query_cache import load_cache, save_cache

EVAL_SET_PATH = Path(__file__).parent / "eval" / "retrieval_eval_set.json"
RESULTS_DIR = Path(__file__).parent / "eval" / "results"
CACHE_PATH = Path(__file__).parent / "eval" / "query_embedding_cache.json"


def load_eval_set() -> list:
    """Read the hand-labeled question set. Each entry: query,
    expected_source_file, expected_header_contains (nullable — only needed
    for source files that produce more than one parent chunk, like
    note2-nested.md), difficulty (exact | paraphrase, for breakdown)."""
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def resolve_promoted(promoted: list, parents: list) -> list:
    """Turn search()'s raw (parent_id, score) pairs into something we can
    actually compare against the eval set's expected_source_file /
    expected_header_contains — resolves each parent_id to its real
    source_file and header_path via the same load_parents() lookup
    hybrid_search.py itself uses. Pure function (plain data in, plain data
    out) — easy to unit test with a synthetic parents list, no FAISS/AWS
    involved. See tests/test_evaluate_retrieval.py."""
    resolved = []
    for parent_id, score in promoted:
        parent = parents[parent_id]
        resolved.append({
            "parent_id": parent_id,
            "source_file": parent["source_file"],
            "header_path": parent["header_path"],
            "score": score,
        })
    return resolved


def find_expected_rank(resolved: list, expected_source_file: str, expected_header_contains) -> int:
    """1-indexed rank of the first resolved result matching the expected
    answer, or None if it never appears in the returned results at all.
    expected_header_contains, when set, is matched as a case-insensitive
    substring against header_path — needed to disambiguate files (like
    note2-nested.md) that produce more than one distinct parent chunk, so
    "the right file" alone isn't precise enough to call it correct.

    Pure function — see tests/test_evaluate_retrieval.py."""
    for rank, item in enumerate(resolved, start=1):
        if item["source_file"] != expected_source_file:
            continue
        if expected_header_contains and expected_header_contains.lower() not in (item["header_path"] or "").lower():
            continue
        return rank
    return None


def reciprocal_rank(rank) -> float:
    """1/rank if found, 0.0 if the expected answer never appeared at all —
    the standard MRR per-query score. Pure function."""
    return 1.0 / rank if rank else 0.0


def run_eval() -> dict:
    """Run every question in the eval set through search(), score each with
    Hit Rate + reciprocal rank, print a per-question line plus an overall
    summary (broken down by difficulty), and return the full result record
    (also saved to eval/results/ by main() below) so future runs can be
    compared against past ones.

    Opts into the query embedding cache (query_cache.py) — loaded once here,
    passed into every search() call, and saved once after the loop (below),
    not per-question."""
    eval_set = load_eval_set()
    parents = load_parents()
    cache = load_cache(CACHE_PATH)
    cache_size_before = len(cache)

    per_question = []
    for case in eval_set:
        promoted = search(case["query"], verbose=False, cache=cache)
        resolved = resolve_promoted(promoted, parents)
        rank = find_expected_rank(resolved, case["expected_source_file"], case.get("expected_header_contains"))
        hit = rank is not None
        rr = reciprocal_rank(rank)

        per_question.append({
            "id": case["id"],
            "query": case["query"],
            "difficulty": case["difficulty"],
            "expected_source_file": case["expected_source_file"],
            "expected_header_contains": case.get("expected_header_contains"),
            "returned": [f"{r['source_file']} :: {r['header_path'] or '(no header)'}" for r in resolved],
            "hit": hit,
            "rank": rank,
            "reciprocal_rank": rr,
        })

        status = f"HIT  (rank {rank})" if hit else "MISS"
        print(f"[{case['difficulty']:10s}] {status:14s} {case['id']:20s} {case['query']!r}")

    # Cache hits = questions that DIDN'T grow the cache (their key was
    # already present); new entries = cache misses that got embedded and
    # cached just now. Saved once here (not per-question) so a run that
    # adds several new entries only hits disk once — see query_cache.py.
    new_cache_entries = len(cache) - cache_size_before
    cache_hits = len(eval_set) - new_cache_entries
    save_cache(cache, CACHE_PATH)
    print(f"Query embedding cache: {cache_hits} hit(s), {new_cache_entries} new entry(ies) "
          f"cached (saved {cache_hits} HyDE+embed Bedrock call pair(s) this run)")

    hit_rate = sum(q["hit"] for q in per_question) / len(per_question)
    mrr = sum(q["reciprocal_rank"] for q in per_question) / len(per_question)

    # Breakdown by difficulty — this is the axis that actually motivated
    # HyDE in the first place (exact-keyword vs paraphrased queries scoring
    # very differently), so it's worth seeing separately, not just averaged
    # away into one overall number.
    by_difficulty = {}
    for difficulty in sorted({q["difficulty"] for q in per_question}):
        subset = [q for q in per_question if q["difficulty"] == difficulty]
        by_difficulty[difficulty] = {
            "hit_rate": sum(q["hit"] for q in subset) / len(subset),
            "mrr": sum(q["reciprocal_rank"] for q in subset) / len(subset),
            "count": len(subset),
        }

    print(f"\n{'='*60}")
    print(f"Overall — Hit Rate@{len(per_question) and 'top_k'}: {hit_rate:.3f}   MRR: {mrr:.3f}   ({len(per_question)} questions)")
    for difficulty, stats in by_difficulty.items():
        print(f"  {difficulty:10s} — Hit Rate: {stats['hit_rate']:.3f}   MRR: {stats['mrr']:.3f}   ({stats['count']} questions)")
    print(f"{'='*60}\n")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall": {"hit_rate": hit_rate, "mrr": mrr, "count": len(per_question)},
        "by_difficulty": by_difficulty,
        "per_question": per_question,
    }


def main():
    result = run_eval()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
