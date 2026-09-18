"""
query_cache.py — cache the final EMBEDDING VECTOR for a query, keyed on
everything that can actually change what that vector is, so re-running the
same query only re-calls Bedrock when something that matters really changed.

Problem statement: step04_hybrid_search.py's vector_rank() calls
transform_query_for_embedding() — under HyDE (the default), that's a real
Claude Haiku generation call, and generation is NOT deterministic, so the
exact same query text can come back with a differently-phrased hypothetical
passage on two separate runs — and then embed_text() (a real Titan Embed
call), with no memoization anywhere. For a one-off interactive query
(`python step04_hybrid_search.py <query>`) that's exactly right — you want HyDE's
natural variation and a fresh call every time. But evaluate_retrieval.py
re-runs the SAME fixed question set after every retrieval-affecting
pipeline change (CANDIDATE_POOL, the reranker, LOCAL_MIN_SCORE, chunk
size — see docs/evaluation-strategies.md's re-run policy), and NONE of
those changes touch the query-embedding step at all. Re-embedding (and
re-generating a fresh HyDE passage) for every question on every single eval
re-run burns real Bedrock calls for work whose result can't have changed —
and HyDE's non-determinism becomes a confound: a Hit Rate shift between two
eval runs could be from the pipeline change actually being tested, or just
from HyDE happening to phrase things differently that time. Caching removes
that confound for any change that doesn't touch embedding at all.

Solution: cache the FINAL vector (post-transform, post-embed) per query,
keyed on everything that determines it — the raw query text, the query
transform mode (none|hyde), the embedding model id, and (only when the mode
is "hyde") the HyDE model id, since that's irrelevant otherwise. Changing
any one of those invalidates just that entry, not the whole cache.

This is opt-in via the `cache` parameter added to step04_hybrid_search.py's
vector_rank()/search() — the interactive CLI path stays uncached by
default (cache=None), so ad hoc querying still gets HyDE's real variation
and never serves a stale vector. evaluate_retrieval.py is the one caller
that opts in, since repeated fixed queries are exactly what this is for.

Storage: a plain JSON file (RAG\\eval\\query_embedding_cache.json) —
consistent with the other JSON/JSONL artifacts already used in this
project, no new dependency, human-readable, and easy to delete by hand to
force a full re-embed if ever needed (e.g. after changing HyDE's prompt
template, which isn't itself part of the cache key).
"""

import json
from pathlib import Path


def cache_key(query: str, transform_mode: str, embed_model_id: str, hyde_model_id: str = "") -> str:
    """Build the cache key for one query's final embedding vector.

    hyde_model_id is deliberately IGNORED (forced to "") when
    transform_mode != "hyde" — the HyDE model can't have affected a "none"
    mode result, so two "none" entries must collide into the same key
    regardless of whatever HYDE_MODEL_ID happens to be configured at the
    time, and a "none" entry must never get invalidated by an unrelated
    HyDE model change. Pure function — see tests/test_query_cache.py."""
    effective_hyde_id = hyde_model_id if transform_mode == "hyde" else ""
    return f"{transform_mode}|{embed_model_id}|{effective_hyde_id}|{query}"


def load_cache(path: Path) -> dict:
    """Read the on-disk cache, or return an empty one if it doesn't exist
    yet (first run) — never an error, just nothing to reuse yet."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(cache: dict, path: Path) -> None:
    """Write the whole cache back out. Called once per evaluate_retrieval.py
    run (not once per query) — see main() there — so a run that adds several
    new entries only hits disk once at the end."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def get_cached_vector(cache: dict, key: str):
    """Returns the cached vector (a plain list[float], same shape
    embed_text() returns) or None on a miss."""
    return cache.get(key)


def set_cached_vector(cache: dict, key: str, vector: list) -> None:
    """Mutates `cache` in place — caller is responsible for save_cache()
    afterward; kept separate from that so a caller can batch many sets into
    one disk write instead of saving after every single query."""
    cache[key] = vector
