"""
test_hybrid_search.py — unit tests for the pure-logic pieces of hybrid_search.py:
apply_min_score() (the relevance-threshold filter) and reciprocal_rank_fusion().

Both take plain Python data in and return plain Python data out — no AWS call,
no cross-encoder model download, no FAISS index needed — so these run in well
under a second and don't depend on any live retrieval pipeline. Importing the
module does still run embed.py's module-level `REGION = os.environ["AWS_REGION"]`
lookup, so an .env with AWS_REGION set (which you already have) needs to be
discoverable from wherever pytest runs — run from the RAG/ directory.

Run: pip install pytest && pytest   (from RAG/)
"""

from hybrid_search import apply_min_score, reciprocal_rank_fusion


# ---- apply_min_score ----
# This is the function born directly from Sept 13's live test: query "git tag"
# returned one genuinely relevant chunk (+4.90) and two confidently irrelevant
# ones (-9.74, -10.51) that fixed top-k was padding the results with.

def test_apply_min_score_none_keeps_old_fixed_top_k_behavior():
    """min_score=None must reproduce the original behavior exactly: always
    return top_k items, no matter how bad the trailing scores are."""
    ranked = [(0, 5.0), (1, -9.0), (2, -11.0), (3, -20.0)]
    assert apply_min_score(ranked, None, top_k=3) == [(0, 5.0), (1, -9.0), (2, -11.0)]


def test_apply_min_score_drops_below_threshold():
    """The exact shape observed live: one clearly relevant match, the rest
    confidently negative — threshold should keep only the first."""
    ranked = [(0, 4.8959), (1, -9.7445), (2, -10.5090)]
    assert apply_min_score(ranked, min_score=0.0, top_k=3) == [(0, 4.8959)]


def test_apply_min_score_can_return_empty():
    """No candidate clears the bar -> empty list, not padded nonsense fed
    downstream to a generation step."""
    ranked = [(0, -1.0), (1, -2.0)]
    assert apply_min_score(ranked, min_score=0.0, top_k=3) == []


def test_apply_min_score_still_caps_at_top_k():
    """A threshold only ever REMOVES candidates — it must never return more
    than top_k even when every candidate clears the bar."""
    ranked = [(0, 5.0), (1, 4.0), (2, 3.0), (3, 2.0)]
    assert apply_min_score(ranked, min_score=0.0, top_k=2) == [(0, 5.0), (1, 4.0)]


def test_apply_min_score_boundary_is_inclusive():
    """A score exactly AT the threshold is kept (>=, not >) — worth pinning
    down explicitly since it's an easy comparison to get backwards."""
    ranked = [(0, 0.0)]
    assert apply_min_score(ranked, min_score=0.0, top_k=3) == [(0, 0.0)]


# ---- reciprocal_rank_fusion ----

def test_rrf_favors_items_ranked_well_in_both_lists():
    """Item 1 is #1 in both BM25 and vector rankings -> should come out on
    top of the fused ranking, ahead of anything that only did well once."""
    bm25_ranked = [1, 2, 3]
    vector_ranked = [1, 3, 2]
    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked])
    assert fused[0][0] == 1


def test_rrf_gives_partial_credit_to_single_list_matches():
    """An item present in only ONE ranked list still appears in the fused
    result (not dropped) — that's the point of fusing two independent
    signals: neither alone gets to silently veto a candidate."""
    bm25_ranked = [5]
    vector_ranked = [6]
    fused = dict(reciprocal_rank_fusion([bm25_ranked, vector_ranked]))
    assert 5 in fused and 6 in fused
    # both were rank 0 (i.e. #1) in their own single list, so equal RRF score
    assert fused[5] == fused[6]
