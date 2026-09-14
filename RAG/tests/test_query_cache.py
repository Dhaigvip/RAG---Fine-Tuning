"""
test_query_cache.py — unit tests for query_cache.py's key-building and
load/save round trip. All pure functions plus simple file I/O against
pytest's tmp_path fixture — no real Bedrock/FAISS call anywhere here.

What's worth testing: does cache_key() actually distinguish everything it
needs to (query text, transform mode, embed model, HyDE model) while
correctly IGNORING the HyDE model when mode="none" (the one subtle rule in
this module — see cache_key()'s docstring for why); does load_cache()
handle a missing file cleanly; does a save+load round trip preserve data.

Run: pytest   (from RAG/)
"""

from query_cache import cache_key, load_cache, save_cache, get_cached_vector, set_cached_vector


def test_cache_key_distinguishes_different_queries():
    key_a = cache_key("git tag", "none", "amazon.titan-embed-text-v2:0")
    key_b = cache_key("git stash", "none", "amazon.titan-embed-text-v2:0")
    assert key_a != key_b


def test_cache_key_distinguishes_transform_mode():
    """Same query, same embed model, different mode (none vs hyde) — since
    the ACTUAL embedding input differs completely between the two, these
    must never collide."""
    key_none = cache_key("git tag", "none", "amazon.titan-embed-text-v2:0")
    key_hyde = cache_key("git tag", "hyde", "amazon.titan-embed-text-v2:0", "some-hyde-model")
    assert key_none != key_hyde


def test_cache_key_distinguishes_embed_model():
    key_a = cache_key("git tag", "none", "amazon.titan-embed-text-v2:0")
    key_b = cache_key("git tag", "none", "some-other-embed-model")
    assert key_a != key_b


def test_cache_key_distinguishes_hyde_model_when_mode_is_hyde():
    """Changing the HyDE model changes what gets generated, so it must
    change the key — but only in hyde mode, see the next test."""
    key_a = cache_key("git tag", "hyde", "amazon.titan-embed-text-v2:0", "claude-3-haiku")
    key_b = cache_key("git tag", "hyde", "amazon.titan-embed-text-v2:0", "claude-haiku-4.5")
    assert key_a != key_b


def test_cache_key_ignores_hyde_model_when_mode_is_none():
    """This is the important one: HYDE_MODEL_ID has zero effect on a "none"
    mode result (HyDE never runs), so two "none" entries must produce the
    SAME key regardless of what hyde_model_id happens to be passed in —
    otherwise an unrelated HyDE model change would wrongly invalidate
    every cached "none" entry too."""
    key_a = cache_key("git tag", "none", "amazon.titan-embed-text-v2:0", "claude-3-haiku")
    key_b = cache_key("git tag", "none", "amazon.titan-embed-text-v2:0", "claude-haiku-4.5")
    assert key_a == key_b


def test_load_cache_returns_empty_dict_for_missing_file(tmp_path):
    """First-ever run, no cache file on disk yet — must return {}, not
    raise."""
    missing_path = tmp_path / "does_not_exist.json"
    assert load_cache(missing_path) == {}


def test_save_and_load_cache_round_trip(tmp_path):
    """A saved cache, read back, must be byte-for-byte the same data —
    also confirms save_cache() creates the parent directory (eval/ may not
    exist yet on a totally fresh checkout)."""
    cache_path = tmp_path / "nested" / "query_embedding_cache.json"
    original = {"none|model-a||git tag": [0.1, 0.2, 0.3]}

    save_cache(original, cache_path)
    assert cache_path.exists()

    loaded = load_cache(cache_path)
    assert loaded == original


def test_get_and_set_cached_vector():
    cache = {}
    key = cache_key("git tag", "none", "amazon.titan-embed-text-v2:0")

    assert get_cached_vector(cache, key) is None  # miss before anything is set

    set_cached_vector(cache, key, [0.1, 0.2, 0.3])
    assert get_cached_vector(cache, key) == [0.1, 0.2, 0.3]  # hit after setting
