"""
test_evaluate_retrieval.py — unit tests for evaluate_retrieval.py's scoring
logic (resolve_promoted, find_expected_rank, reciprocal_rank), all pure
functions with synthetic data — no FAISS index, no AWS call, no real eval
set file needed. What's worth testing here is the SCORING logic itself:
does rank get computed correctly, does the header_path disambiguation
actually disambiguate, does a miss correctly score 0 rather than crashing.

Run: pytest   (from RAG/)
"""

from evaluate_retrieval import resolve_promoted, find_expected_rank, reciprocal_rank


# A small synthetic "parents" list, positionally indexed exactly like the
# real chunks.jsonl load_parents() returns — parent_id is just the index.
FAKE_PARENTS = [
    {"source_file": "note1-flat.md", "header_path": None, "text": "..."},
    {"source_file": "note2-nested.md", "header_path": "System Design Notes > Load Balancing > Layer 4 vs Layer 7", "text": "..."},
    {"source_file": "note2-nested.md", "header_path": "System Design Notes > Caching Strategies", "text": "..."},
]


def test_resolve_promoted_maps_parent_id_to_source_and_header():
    """resolve_promoted() must look up each parent_id's real source_file
    and header_path via the parents list, carrying the score through
    unchanged."""
    promoted = [(1, 5.5), (0, 2.0)]
    resolved = resolve_promoted(promoted, FAKE_PARENTS)
    assert resolved == [
        {"parent_id": 1, "source_file": "note2-nested.md", "header_path": "System Design Notes > Load Balancing > Layer 4 vs Layer 7", "score": 5.5},
        {"parent_id": 0, "source_file": "note1-flat.md", "header_path": None, "score": 2.0},
    ]


def test_find_expected_rank_matches_on_source_file_alone_when_no_header_filter():
    """When expected_header_contains is None, matching on source_file alone
    is enough — this is the note1/note4/note5 case (single parent per
    file in the eval set)."""
    resolved = [
        {"parent_id": 2, "source_file": "note2-nested.md", "header_path": "Caching Strategies", "score": 1.0},
        {"parent_id": 0, "source_file": "note1-flat.md", "header_path": None, "score": 0.5},
    ]
    assert find_expected_rank(resolved, "note1-flat.md", None) == 2


def test_find_expected_rank_disambiguates_via_header_substring():
    """This is the important one: note2-nested.md produces TWO distinct
    parent chunks (Layer 4 vs Layer 7, and Caching Strategies). Matching on
    source_file alone would call a Caching-Strategies result correct for a
    Layer-4-vs-7 question, which is wrong. expected_header_contains must
    actually filter that out."""
    resolved = [
        {"parent_id": 2, "source_file": "note2-nested.md", "header_path": "System Design Notes > Caching Strategies", "score": 3.0},
        {"parent_id": 1, "source_file": "note2-nested.md", "header_path": "System Design Notes > Load Balancing > Layer 4 vs Layer 7", "score": 1.0},
    ]
    # Wrong section ranked first — must NOT count as a hit for the L4-vs-7 question.
    assert find_expected_rank(resolved, "note2-nested.md", "Layer 4 vs Layer 7") == 2
    # But it must still count as a hit for the Caching question, at rank 1.
    assert find_expected_rank(resolved, "note2-nested.md", "Caching") == 1


def test_find_expected_rank_returns_none_on_a_genuine_miss():
    """The expected source never appearing at all — e.g. the relevance
    threshold filtered out everything, or retrieval just got it wrong —
    must return None, not raise or silently pick something wrong."""
    resolved = [
        {"parent_id": 0, "source_file": "note1-flat.md", "header_path": None, "score": 1.0},
    ]
    assert find_expected_rank(resolved, "note4-many-small.md", None) is None


def test_find_expected_rank_on_empty_results():
    """search() returns [] when nothing passed the relevance threshold at
    all (see hybrid_search.py's search()) — resolve_promoted([], parents)
    is just [], and find_expected_rank must handle that as a clean miss,
    not an error."""
    assert find_expected_rank([], "note1-flat.md", None) is None


def test_reciprocal_rank_of_a_hit_and_a_miss():
    """1/rank for a hit, 0.0 (not an error, not None) for a miss — this is
    what run_eval() averages directly into MRR."""
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(3) == 1.0 / 3
    assert reciprocal_rank(None) == 0.0
