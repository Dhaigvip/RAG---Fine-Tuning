"""
test_generate.py — unit tests for step05_generate.py's pure functions: resolve_sources,
build_prompt, parse_citations, detect_refusal. All plain-data-in/plain-data-out
(or plain-text-in/plain-data-out) — no Bedrock call, no FAISS, no search()
involved anywhere in this file.

What's worth testing here: does resolve_sources() number sources correctly and
in search()'s existing rank order (not resorted); does build_prompt() actually
include every source and the question; does parse_citations() correctly split
valid vs out-of-range citation numbers, including edge cases (no citations,
duplicate citations, citation number 0); does detect_refusal() match
case-insensitively and NOT false-positive on an answer that merely mentions
similar words without the actual phrase.

Run: pytest   (from RAG/)
"""

from step05_generate import resolve_sources, build_prompt, parse_citations, detect_refusal, REFUSAL_PHRASE


# A small synthetic "parents" list, positionally indexed exactly like the
# real chunks.jsonl load_parents() returns — parent_id is just the index.
# Mirrors the FAKE_PARENTS convention already used in test_evaluate_retrieval.py.
FAKE_PARENTS = [
    {"source_file": "note1-flat.md", "header_path": None, "text": "Docker build cache gotcha text..."},
    {"source_file": "note2-nested.md", "header_path": "System Design Notes > Load Balancing > Layer 4 vs Layer 7", "text": "L4 vs L7 text..."},
    {"source_file": "note2-nested.md", "header_path": "System Design Notes > Caching Strategies", "text": "Caching text..."},
]


def test_resolve_sources_numbers_in_search_rank_order():
    """resolve_sources() must number sources 1, 2, 3... in the SAME order
    search() already ranked them (promoted's order), never re-sorted by
    score or anything else — that order is what the prompt shows the model
    and what parse_citations() checks citations against, so it must be
    deterministic and match promoted exactly."""
    promoted = [(1, 5.5), (0, 2.0)]  # parent_id 1 first, then parent_id 0
    sources = resolve_sources(promoted, FAKE_PARENTS)
    assert sources == [
        {"index": 1, "parent_id": 1, "source_file": "note2-nested.md",
         "header_path": "System Design Notes > Load Balancing > Layer 4 vs Layer 7",
         "text": "L4 vs L7 text...", "score": 5.5},
        {"index": 2, "parent_id": 0, "source_file": "note1-flat.md",
         "header_path": None, "text": "Docker build cache gotcha text...", "score": 2.0},
    ]


def test_resolve_sources_on_empty_promoted_list():
    """generate_answer() only calls this after already checking promoted is
    non-empty, but resolve_sources() itself should still handle [] cleanly
    (return []) rather than assume a caller always guards it."""
    assert resolve_sources([], FAKE_PARENTS) == []


def test_build_prompt_includes_every_source_and_the_question():
    sources = resolve_sources([(0, 1.0), (2, 0.5)], FAKE_PARENTS)
    prompt = build_prompt("How do I fix Docker cache issues?", sources)

    assert "[1] note1-flat.md" in prompt
    assert "Docker build cache gotcha text..." in prompt
    assert "[2] note2-nested.md — System Design Notes > Caching Strategies" in prompt
    assert "Caching text..." in prompt
    assert "QUESTION: How do I fix Docker cache issues?" in prompt


def test_build_prompt_omits_header_label_when_header_path_is_none():
    """note1-flat.md has header_path=None (see FAKE_PARENTS and the real
    note1/note4/note5 pattern documented in evaluation-strategies.md) — the
    label for that source must not print a stray '— None' or similar."""
    sources = resolve_sources([(0, 1.0)], FAKE_PARENTS)
    prompt = build_prompt("irrelevant", sources)
    assert "[1] note1-flat.md\n" in prompt
    assert "None" not in prompt


def test_parse_citations_finds_valid_citations():
    result = parse_citations("Docker caches layers [1] unless you use --no-cache [1].", num_sources=2)
    assert result == {"found": [1], "invalid": []}


def test_parse_citations_deduplicates_repeated_citations():
    """The same source cited twice in one answer should appear once in
    'found', not twice — 'found' is the set of sources actually used, not a
    count of citation occurrences."""
    result = parse_citations("[2] and again [2] and also [1].", num_sources=2)
    assert result == {"found": [1, 2], "invalid": []}


def test_parse_citations_flags_out_of_range_numbers():
    """Only 2 sources were provided, but the model cited [3] — a fabricated
    or miscounted citation. Must be flagged in 'invalid', not silently
    accepted as if source 3 existed."""
    result = parse_citations("According to [1] and [3], ...", num_sources=2)
    assert result == {"found": [1, 3], "invalid": [3]}


def test_parse_citations_flags_citation_zero_as_invalid():
    """Source numbering is 1-based (see resolve_sources) — [0] is never a
    real source number, so it must land in 'invalid' too."""
    result = parse_citations("See [0] for details.", num_sources=2)
    assert result == {"found": [0], "invalid": [0]}


def test_parse_citations_on_answer_with_no_citations_at_all():
    """A refusal answer (see REFUSAL_PHRASE) has no citations at all — must
    return empty lists cleanly, not error on no matches."""
    result = parse_citations(REFUSAL_PHRASE, num_sources=3)
    assert result == {"found": [], "invalid": []}


def test_detect_refusal_matches_case_insensitively():
    assert detect_refusal(REFUSAL_PHRASE.upper()) is True
    assert detect_refusal(REFUSAL_PHRASE.lower()) is True


def test_detect_refusal_true_when_phrase_is_embedded_in_more_text():
    """The model might add nothing else per the system prompt's instruction
    ('respond with EXACTLY this sentence and nothing else'), but detection
    itself should not depend on the phrase being the ENTIRE answer — a
    substring match is deliberately more forgiving than an exact-equality
    check, in case the model adds stray whitespace or punctuation around it."""
    assert detect_refusal(f"  {REFUSAL_PHRASE}  ") is True


def test_detect_refusal_false_on_a_normal_grounded_answer():
    """Must NOT false-positive just because an answer happens to mention
    'information' or 'notes' — only the actual fixed phrase counts."""
    answer = "Docker caches build layers [1]; use --no-cache to force a rebuild [1]."
    assert detect_refusal(answer) is False


def test_detect_refusal_false_on_answer_merely_expressing_uncertainty():
    """A model hedging in its own words ('I'm not entirely sure...') is a
    DIFFERENT, unhandled failure mode from a proper refusal — it must not be
    mistaken for one just because it sounds similar. This is exactly the
    documented gap in docs/generation-strategies.md: detect_refusal() only
    catches the model correctly using the fixed phrase, nothing else."""
    answer = "I'm not entirely sure, but it might be related to caching."
    assert detect_refusal(answer) is False
