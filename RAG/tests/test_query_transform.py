"""
test_query_transform.py — unit tests for transform_query_for_embedding()'s
dispatch + fallback logic, WITHOUT calling Bedrock. generate_hyde_document()
itself isn't tested here (it's a thin wrapper around one bedrock.invoke_model
call — there's nothing to unit test there beyond "does boto3 get called
right," which isn't worth mocking deeply); what's actually worth testing is
the decision logic: does "none" mode skip generation entirely, does "hyde"
mode use the generated text, and — the important one — does a generation
failure fall back to the raw query instead of blowing up search()?

Each test directly sets module attributes (QUERY_TRANSFORM,
generate_hyde_document) rather than environment variables, since
QUERY_TRANSFORM is read once at import time — this is the same pattern
step04_hybrid_search.py's own tests would use for monkeypatching its module-level
functions.

Run: pytest   (from RAG/)
"""

import query_transform as qt


def test_none_mode_returns_raw_query_untouched():
    """QUERY_TRANSFORM="none" must never call the LLM at all — plain
    passthrough, byte-for-byte the same query string back."""
    original_mode = qt.QUERY_TRANSFORM
    try:
        qt.QUERY_TRANSFORM = "none"
        assert qt.transform_query_for_embedding("git tag") == "git tag"
    finally:
        qt.QUERY_TRANSFORM = original_mode


def test_hyde_mode_uses_the_generated_document():
    """QUERY_TRANSFORM="hyde" should return whatever generate_hyde_document()
    produced, not the original query."""
    original_mode = qt.QUERY_TRANSFORM
    original_fn = qt.generate_hyde_document
    try:
        qt.QUERY_TRANSFORM = "hyde"
        qt.generate_hyde_document = lambda query: "a fake hypothetical answer"
        assert qt.transform_query_for_embedding("git tag") == "a fake hypothetical answer"
    finally:
        qt.QUERY_TRANSFORM = original_mode
        qt.generate_hyde_document = original_fn


def test_hyde_mode_falls_back_to_raw_query_on_any_failure():
    """This is the important one: if generation raises for ANY reason
    (throttling, an SCP block, network — this project has direct precedent
    for the SCP case, see reranking-strategies.md), the raw query must still
    be returned so vector_rank() degrades gracefully instead of search()
    crashing outright."""
    original_mode = qt.QUERY_TRANSFORM
    original_fn = qt.generate_hyde_document
    try:
        qt.QUERY_TRANSFORM = "hyde"

        def _boom(query):
            raise RuntimeError("simulated Bedrock failure")

        qt.generate_hyde_document = _boom
        assert qt.transform_query_for_embedding("git tag") == "git tag"
    finally:
        qt.QUERY_TRANSFORM = original_mode
        qt.generate_hyde_document = original_fn
