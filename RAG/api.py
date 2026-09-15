"""
api.py — a minimal FastAPI wrapper around generate.py's generate_answer(),
so this RAG pipeline is callable over HTTP instead of only as a one-shot CLI
script (python generate.py "<question>"). This is the "assemble end-to-end
pipeline (simple FastAPI endpoint)" item from the Sept 15-16 plan.

See docs/api-strategies.md for the full landscape and the real design
decisions made here: FastAPI vs Flask, sync `def` vs `async def` (matters —
see below), why the FAISS index/parents aren't cached across requests yet
(a real, documented gap, not an oversight), and the error-handling approach.

Requires: pip install fastapi "uvicorn[standard]"
(tests additionally need: pip install httpx — FastAPI's TestClient depends on it)

Usage:
    uvicorn api:app --reload
        # from RAG\, starts a local dev server at http://127.0.0.1:8000
        # interactive docs (try it in the browser): http://127.0.0.1:8000/docs
    Then, e.g. from a second terminal:
    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"how do I clean up branches and stashes\"}"
    curl http://127.0.0.1:8000/health
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generate import generate_answer

app = FastAPI(
    title="Personal Notes RAG API",
    description=(
        "Ask a question, get an answer grounded in the personal-notes corpus, "
        "with citations back to the source notes and a refusal when the notes "
        "don't actually cover the question."
    ),
    version="0.1.0",
)


class AskRequest(BaseModel):
    """Request body for POST /ask. Pydantic validates this automatically —
    a request missing "query", or sending an empty string, gets a clean 422
    error from FastAPI before generate_answer() (and any real Bedrock/FAISS
    call) ever runs — see tests/test_api.py's validation tests."""
    query: str = Field(..., min_length=1, description="The question to ask the notes corpus.")


class Source(BaseModel):
    """One retrieved source, as generate.py's generate_answer() already
    summarizes it (see that file's docstring) — this mirrors that shape
    exactly rather than inventing a new one."""
    index: int
    source_file: str
    header_path: Optional[str]
    score: float


class AskResponse(BaseModel):
    """Mirrors generate_answer()'s return dict field-for-field — see
    generate.py's docstring for what each field means. Declaring this
    explicitly (instead of just returning the raw dict) is what lets
    FastAPI generate the /docs schema and validate the response shape
    automatically.

    citations is left as a plain dict rather than its own model: it's {}
    on the no_retrieval refusal path (nothing to cite) and
    {"found": [...], "invalid": [...]} otherwise — two genuinely different
    shapes for the same field, not worth a Union model for this endpoint."""
    query: str
    refused: bool
    reason: Optional[str]
    answer: str
    sources: list[Source]
    citations: dict


@app.get("/health")
def health():
    """Liveness check — no AWS/FAISS call at all, just confirms the process
    is up and can serve requests. Deliberately does NOT check whether
    Bedrock/FAISS are actually reachable (a real readiness check that
    verifies downstream dependencies is a further step, not this one) —
    that would make /health itself slow and AWS-dependent, defeating the
    point of a liveness check."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """The one real endpoint. Declared as plain `def`, not `async def` —
    see docs/api-strategies.md ("sync def, not async def") for the full
    reasoning: everything generate_answer() calls underneath (boto3, FAISS,
    sentence-transformers) is synchronous/blocking, and FastAPI runs a sync
    `def` endpoint in a thread pool automatically, so one slow Bedrock call
    here doesn't freeze every other concurrent request the way it would
    inside an `async def` on the event loop.

    verbose=False here, unlike the CLI's own default (verbose=True) — this
    is a server process that may serve many requests; the CLI's per-request
    debug printing (BM25/vector/RRF candidate lists, the HyDE hypothetical
    doc) is genuinely useful for one-off manual testing but would just be
    noise in a server's stdout under real traffic."""
    try:
        return generate_answer(request.query, verbose=False)
    except Exception as e:
        # generate_answer() only raises from call_generation_model() after
        # exhausting its retry budget (see generate.py) — a real Bedrock
        # outage/throttling situation, not a bug in this specific request.
        # 503 (service temporarily unavailable) is the honest status for
        # that. Only the exception TYPE name is included, never str(e) —
        # internal error text (which could contain request/response detail
        # from boto3) never reaches the caller. See docs/api-strategies.md
        # for why this is deliberately coarse (doesn't yet distinguish a
        # transient outage from a setup/config error like a missing index).
        raise HTTPException(
            status_code=503,
            detail=f"Generation temporarily unavailable: {type(e).__name__}",
        ) from e
