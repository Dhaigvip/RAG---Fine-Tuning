"""
test_api.py — endpoint-level tests for api.py, using FastAPI's TestClient
(calls the app in-process — no real server, no real network). generate_answer()
is mocked (unittest.mock.patch) in every test here: this file tests the API
LAYER (request validation, response shape, error translation), not
generate_answer()'s own logic — that's already covered by
tests/test_generate.py's 15 pure-function tests. No real Bedrock/FAISS call
anywhere in this file.

Requires: pip install httpx (FastAPI's TestClient depends on it)
Run: pytest   (from RAG/)
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_generate_answer_result_on_success():
    """A normal, grounded answer — api.py should pass the request through to
    generate_answer() and return its result shape unchanged, with
    verbose=False (see api.py's ask() docstring for why)."""
    fake_result = {
        "query": "how do I clean up branches and stashes",
        "refused": False,
        "reason": None,
        "answer": "Use `git branch -d` [1] and `git stash pop` [1].",
        "sources": [{"index": 1, "source_file": "note4-many-small.md", "header_path": "Quick Reference", "score": 0.5}],
        "citations": {"found": [1], "invalid": []},
    }
    with patch("api.generate_answer", return_value=fake_result) as mock_generate:
        response = client.post("/ask", json={"query": "how do I clean up branches and stashes"})

    assert response.status_code == 200
    assert response.json() == fake_result
    mock_generate.assert_called_once_with("how do I clean up branches and stashes", verbose=False)


def test_ask_returns_refusal_result_unchanged():
    """A refusal (empty sources/citations, refused=True) must round-trip
    through the API exactly as generate_answer() returns it — a refusal is
    a normal, valid 200 response, not an error condition."""
    fake_result = {
        "query": "why doesn't my container pick up changes",
        "refused": True,
        "reason": "no_retrieval",
        "answer": "I don't have enough information in these notes to answer that.",
        "sources": [],
        "citations": {},
    }
    with patch("api.generate_answer", return_value=fake_result):
        response = client.post("/ask", json={"query": "why doesn't my container pick up changes"})

    assert response.status_code == 200
    assert response.json() == fake_result


def test_ask_rejects_empty_query_with_422():
    """Pydantic's min_length=1 on AskRequest.query must reject an empty
    string before generate_answer() is ever called (a wasted Bedrock/FAISS
    round trip for a request that was never valid) — request validation,
    not application logic."""
    with patch("api.generate_answer") as mock_generate:
        response = client.post("/ask", json={"query": ""})

    assert response.status_code == 422
    mock_generate.assert_not_called()


def test_ask_rejects_missing_query_field_with_422():
    with patch("api.generate_answer") as mock_generate:
        response = client.post("/ask", json={})

    assert response.status_code == 422
    mock_generate.assert_not_called()


def test_ask_translates_generation_failure_to_503_without_leaking_internals():
    """generate_answer() raises after exhausting retries on a real Bedrock
    outage (see generate.py's call_generation_model()) — the API must turn
    that into a clean 503 naming only the exception TYPE, never leaking the
    raw exception message (which could carry internal boto3/AWS detail) to
    the caller. See api.py's ask() and docs/api-strategies.md."""
    with patch("api.generate_answer", side_effect=RuntimeError("exhausted retries calling Bedrock")):
        response = client.post("/ask", json={"query": "anything"})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "RuntimeError" in detail
    assert "exhausted retries" not in detail
