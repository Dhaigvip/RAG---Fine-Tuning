"""
query_transform.py — HyDE (Hypothetical Document Embeddings) query transformation.

Problem statement: embedding similarity compares the QUERY's vector against
each CHUNK's vector, but a terse question and a well-written answer are
phrased very differently — this project has its own direct evidence of that
gap. From reranking-strategies.md's two-query test, the exact same correct
chunk scored +4.90 for the near-exact-keyword query "git tag" but only +0.53
for the paraphrased query "how do I clean up branches and stashes" — a query
that drifts further from the document's actual wording gets a measurably
weaker match, even when it's still the right answer. That gap happens at the
EMBEDDING/vector-search stage, before reranking ever sees the candidates —
reranking can only re-score whatever vector search + BM25 already surfaced,
it can't rescue a match that never made the candidate pool.

Solution — HyDE (Hypothetical Document Embeddings): instead of embedding the
raw query, ask an LLM to write a short, plausible-sounding (not necessarily
TRUE) passage that would answer it, and embed THAT instead. A generated
answer is phrased like the documents in the corpus are phrased — technical,
declarative, in the vocabulary of an actual answer — so its embedding lands
closer to the real matching chunk than the original terse question's
embedding does. This only replaces the VECTOR-search leg of retrieval;
BM25 keeps using the raw query — see the "why BM25 stays on the raw query"
note in docs/query-transformation-strategies.md.

Landscape of alternatives considered (multi-query fan-out, step-back
prompting, query decomposition) and why HyDE was chosen for this project is
documented in docs/query-transformation-strategies.md — this file is just
the mechanism.

Model choice: anthropic.claude-3-haiku-20240307-v1:0 — confirmed directly
invocable in eu-central-1 with NO cross-region inference profile required
(unlike the newer Claude Haiku 4.5, which needs an "eu."/"global." routed
inference profile in this region — extra IAM surface this project doesn't
need for a task this small). Same boto3/IAM stack already used for
embeddings, no new vendor or credential.

Requires: boto3 (already a dependency via embed.py)
"""

import json
import os

import boto3

from embed import REGION, PROFILE

# "none" disables this entirely (embed the raw query, old behavior).
# "hyde" is the default — see the problem statement above for why it's worth
# the extra LLM call by default rather than opt-in.
QUERY_TRANSFORM = os.environ.get("TOOL_RAG_QUERY_TRANSFORM", "none")

HYDE_MODEL_ID = os.environ.get("TOOL_RAG_HYDE_MODEL", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")
HYDE_MAX_TOKENS = int(os.environ.get("TOOL_RAG_HYDE_MAX_TOKENS", "200"))

HYDE_PROMPT_TEMPLATE = (
    "Write a short passage (2-4 sentences) that would plausibly answer this "
    "question, in the style of a technical note or piece of documentation. "
    "Do not worry about being factually correct or add any hedging - write "
    "confidently as if this were the real answer. This text is only used "
    "internally to improve a search match; it is never shown to a user.\n\n"
    "Question: {query}"
)

_bedrock_runtime = None  # lazy-loaded, same pattern as every other client in this project


def _get_bedrock_runtime():
    global _bedrock_runtime
    if _bedrock_runtime is None:
        session = boto3.Session(profile_name=PROFILE, region_name=REGION) if PROFILE else boto3.Session(region_name=REGION)
        _bedrock_runtime = session.client("bedrock-runtime")
    return _bedrock_runtime


def generate_hyde_document(query: str) -> str:
    """Ask a small, fast Claude model to write a plausible-looking answer to
    the query. The content doesn't need to be TRUE — only phrased the way a
    real answer document would be, so its embedding lands closer to the
    actual matching chunk than the terse original query's embedding does.

    Raises on any Bedrock error (throttling, access denial, an SCP block —
    this project has direct precedent for that last one, see
    reranking-strategies.md). Callers should catch and fall back to the raw
    query rather than let a transformation failure take down the whole
    search — see transform_query_for_embedding()."""
    bedrock = _get_bedrock_runtime()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": HYDE_MAX_TOKENS,
        "messages": [{"role": "user", "content": HYDE_PROMPT_TEMPLATE.format(query=query)}],
    })
    response = bedrock.invoke_model(
        modelId=HYDE_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(response["body"].read())
    return payload["content"][0]["text"].strip()


def transform_query_for_embedding(query: str) -> str:
    """Return whatever text should actually be embedded for the VECTOR leg
    of retrieval (hybrid_search.py's vector_rank calls this). BM25 always
    uses the raw query, completely unaffected by this function — see
    docs/query-transformation-strategies.md.

    Dispatch + graceful fallback in one place: QUERY_TRANSFORM="none" is a
    plain passthrough (old behavior, zero risk). QUERY_TRANSFORM="hyde"
    tries generate_hyde_document(), but ANY failure there (network,
    throttling, an SCP block) falls back to embedding the raw query instead
    of crashing search() outright — the same "degrade, don't crash"
    philosophy already used for reranking's dual-backend design."""
    if QUERY_TRANSFORM != "hyde":
        return query
    try:
        hyde_text = generate_hyde_document(query)
        print(f"  [HyDE] hypothetical doc: {hyde_text[:150].strip()}...")
        return hyde_text
    except Exception as e:
        print(f"  [HyDE] generation failed ({type(e).__name__}: {e}) — falling back to raw query for embedding.")
        return query
