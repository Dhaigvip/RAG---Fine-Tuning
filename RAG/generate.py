"""
generate.py — the GENERATION half of the RAG pipeline: take search()'s
retrieved parent chunks and turn them into an actual answer, grounded in
those chunks, with citations back to which source(s) it drew from, and a
refusal when the retrieved context doesn't actually answer the question.

Problem statement: hybrid_search.py's search() only ever returns retrieved
chunks — nothing in this project has turned a chunk into an answer a person
reads. Two separate risks sit ahead of that: (1) an LLM asked to answer will
answer, from its own general knowledge if it has to, even when the provided
notes don't cover the question — which defeats the point of grounding to a
personal corpus in the first place; (2) a plain prose answer with no
reference to its source can't be checked against the real note if it's
wrong. Concretely, this project already has a real MISS in measured data
(the docker-02 retrieval-eval question) proving that "search() returned
some chunks" does NOT mean "those chunks answer this question" — FINAL_TOP_K
keeps filling from whatever passed the relevance threshold, right or wrong.
See docs/generation-strategies.md for the full landscape and reasoning.

Solution, in order:
1. Retrieve via the existing search() — unchanged, nothing new here.
2. If search() returns [] (nothing passed the relevance threshold), refuse
   immediately — zero generation calls, reusing an already-verified signal.
3. Otherwise, build a prompt with the retrieved parents as clearly numbered,
   bounded context, instructing the model to answer ONLY from that context,
   to cite bracket numbers ([1], [2], ...) for what it drew from, and to
   respond with an exact, fixed refusal phrase if the context doesn't
   actually answer the question.
4. Call Bedrock, get the answer text back.
5. Validate citations against the real source count (flag out-of-range
   numbers, don't hide or auto-correct them) and detect the refusal phrase
   in the output (substring match) so refusal is checkable in code, not
   just visible to a human reading the text.

Requires: boto3 (already a dependency via embed.py / query_transform.py)
Run `python hybrid_search.py <query>` at least once first to confirm
retrieval works before layering generation on top of it.

Usage:
    python generate.py <question text>
"""

import json
import os
import re
import time

import boto3
from botocore.exceptions import ClientError

from embed import REGION, PROFILE
from hybrid_search import search, load_parents

# Separate config from HyDE's model on purpose — see docs/generation-strategies.md
# ("Model — separate config from HyDE"): HyDE's output is throwaway (never
# shown to a user), generation's output IS the user-facing deliverable, so a
# future model upgrade for one shouldn't silently also change the other, even
# though both happen to point at the same model right now.
#
# Updated Sept 15 (explicit standing instruction — supersedes the note this
# replaces): eu.anthropic.claude-haiku-4-5-20251001-v1:0 (Claude Haiku 4.5 via
# the "eu." cross-region inference profile), matching query_transform.py's
# HYDE_MODEL_ID default. This takes on cross-region inference profile IAM
# surface that query_transform.py's ORIGINAL model choice deliberately
# avoided (see that file's docstring) — not yet confirmed working end-to-end
# for generation specifically; watch for an AccessDenied/ValidationException
# here if the profile isn't authorized for this account.
GENERATION_MODEL_ID = os.environ.get("TOOL_RAG_GENERATION_MODEL", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")
GENERATION_MAX_TOKENS = int(os.environ.get("TOOL_RAG_GENERATION_MAX_TOKENS", "500"))
MAX_RETRIES = 5  # same retry budget as embed.py

# The exact phrase the model is instructed to use when retrieved context
# doesn't answer the question. Kept as a single constant so the prompt
# instruction and the detection check can never drift apart from each other.
REFUSAL_PHRASE = "I don't have enough information in these notes to answer that."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered SOURCES provided below — "
    "never your own general knowledge, even if you happen to know the answer. "
    "This is important: these sources are someone's personal notes, and the "
    "whole point is an answer that's traceable back to what they actually "
    "wrote, not a plausible-sounding answer from elsewhere.\n\n"
    "Rules:\n"
    "1. Cite the source number(s) you drew from inline, like [1] or [2], "
    "right next to the claim they support.\n"
    "2. If the sources do not actually answer the question — even if they "
    "are on a related topic — respond with EXACTLY this sentence and "
    "nothing else: \"" + REFUSAL_PHRASE + "\"\n"
    "3. Do not guess, fill gaps from general knowledge, or hedge around an "
    "answer the sources don't support. Refusing is the correct answer when "
    "the sources don't cover it — it is not a failure."
)

_bedrock_runtime = None  # lazy-loaded, same pattern as every other client in this project


def _get_bedrock_runtime():
    global _bedrock_runtime
    if _bedrock_runtime is None:
        session = boto3.Session(profile_name=PROFILE, region_name=REGION) if PROFILE else boto3.Session(region_name=REGION)
        _bedrock_runtime = session.client("bedrock-runtime")
    return _bedrock_runtime


def resolve_sources(promoted: list, parents: list) -> list:
    """Turn search()'s raw (parent_id, score) pairs into numbered source
    records ready to go straight into a prompt: {index, source_file,
    header_path, text, score}. `index` is 1-based and follows search()'s
    existing rank order exactly (no re-sorting) — that's the same number
    both build_prompt() puts in front of each source and parse_citations()
    checks the model's [n] citations against, so the two must always agree
    on what "source N" means.

    Pure function (plain data in, plain data out) — see tests/test_generate.py."""
    sources = []
    for i, (parent_id, score) in enumerate(promoted, start=1):
        parent = parents[parent_id]
        sources.append({
            "index": i,
            "parent_id": parent_id,
            "source_file": parent["source_file"],
            "header_path": parent["header_path"],
            "text": parent["text"],
            "score": score,
        })
    return sources


def build_prompt(query: str, sources: list) -> str:
    """Build the user-message text: each source numbered and labeled with
    its file/header, followed by the actual question. The numbering here is
    what the model is asked to cite back — see resolve_sources()'s docstring
    for why that numbering must stay in lockstep with parse_citations().

    Pure function — see tests/test_generate.py."""
    blocks = []
    for source in sources:
        label = f"[{source['index']}] {source['source_file']}"
        if source["header_path"]:
            label += f" — {source['header_path']}"
        blocks.append(f"{label}\n{source['text']}")

    context = "\n\n".join(blocks)
    return f"SOURCES:\n\n{context}\n\nQUESTION: {query}"


def parse_citations(answer_text: str, num_sources: int) -> dict:
    """Extract every [n] citation the model wrote, split into ones that are
    actually valid (n is a real source number) and ones that aren't (out of
    range — a fabricated or miscounted citation). Doesn't touch the answer
    text itself; this is purely for surfacing a flag, not for rewriting the
    answer — see docs/generation-strategies.md ("Citation validation, not
    enforcement").

    Returns {"found": sorted list of all cited numbers (deduped),
             "invalid": sorted list of cited numbers outside 1..num_sources}.

    Pure function — see tests/test_generate.py."""
    found = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    invalid = {n for n in found if n < 1 or n > num_sources}
    return {"found": sorted(found), "invalid": sorted(invalid)}


def detect_refusal(answer_text: str) -> bool:
    """Case-insensitive substring check for the fixed refusal phrase — this
    is what makes the model's self-reported "the sources don't answer this"
    checkable in code instead of only visible to a human reading the output.
    A known limitation, documented in docs/generation-strategies.md: this
    only catches the model correctly following the refusal instruction — it
    does nothing to catch a case where the model SHOULD have refused but
    hallucinated a confident answer instead. That's the deferred
    faithfulness-check gap, not something this function can fix.

    Pure function — see tests/test_generate.py."""
    return REFUSAL_PHRASE.lower() in answer_text.lower()


def call_generation_model(query: str, sources: list) -> str:
    """The one function in this module that actually calls Bedrock. Retries
    on ThrottlingException with exponential backoff — unlike
    query_transform.py's generate_hyde_document(), which fails fast and
    falls back to the raw query on ANY error. That fail-fast choice makes
    sense for HyDE because it has a safe fallback; generation has no
    fallback — it IS the output — so this follows embed.py's retry pattern
    instead. See docs/generation-strategies.md ("Retry on throttling, unlike
    HyDE") for the full reasoning.

    Raises on any non-throttling error, or after exhausting retries — there
    is no graceful degradation available at this layer; a caller that wants
    one (e.g. falling back to a "generation unavailable" message) should
    catch around this call."""
    bedrock = _get_bedrock_runtime()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": GENERATION_MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_prompt(query, sources)}],
    })

    for attempt in range(MAX_RETRIES):
        try:
            response = bedrock.invoke_model(
                modelId=GENERATION_MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read())
            return payload["content"][0]["text"].strip()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "ThrottlingException" and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  throttled, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("exhausted retries calling Bedrock")


def generate_answer(query: str, verbose: bool = True, cache: dict = None) -> dict:
    """Run the full generate step for one question: retrieve, refuse early
    if retrieval found nothing, otherwise generate + validate citations +
    detect a prompted refusal. Always returns a full result dict, even on
    refusal, so a refusal is inspectable (what WAS retrieved, that just
    didn't answer the question) rather than a bare "no" — useful for
    debugging retrieval gaps like docker-02 later.

    `cache` is passed straight through to search() (see hybrid_search.py /
    query_cache.py) — default None preserves normal fresh-HyDE CLI behavior;
    a future eval harness for generation could opt in the same way
    evaluate_retrieval.py already does for retrieval.

    Returns:
        {
          "query": str,
          "refused": bool,
          "reason": None | "no_retrieval" | "model_declined",
          "answer": str,
          "sources": [ {index, source_file, header_path, score}, ... ],
          "citations": {"found": [...], "invalid": [...]},   # {} on no_retrieval refusal
        }
    """
    promoted = search(query, verbose=verbose, cache=cache)

    if not promoted:
        if verbose:
            print("No chunks passed the relevance threshold — refusing without a generation call.")
        return {
            "query": query,
            "refused": True,
            "reason": "no_retrieval",
            "answer": REFUSAL_PHRASE,
            "sources": [],
            "citations": {},
        }

    parents = load_parents()
    sources = resolve_sources(promoted, parents)

    answer_text = call_generation_model(query, sources)
    citations = parse_citations(answer_text, num_sources=len(sources))
    refused = detect_refusal(answer_text)

    if verbose:
        print(f"\nAnswer:\n{answer_text}\n")
        print(f"Citations found: {citations['found']}"
              + (f"  (INVALID: {citations['invalid']})" if citations["invalid"] else ""))
        print(f"Refused: {refused}")

    # Strip embedding vectors / raw text bulk from the returned source list —
    # callers get enough to know WHAT was retrieved (for debugging a case
    # like docker-02) without re-printing every parent's full text back out.
    source_summary = [
        {"index": s["index"], "source_file": s["source_file"], "header_path": s["header_path"], "score": s["score"]}
        for s in sources
    ]

    return {
        "query": query,
        "refused": refused,
        "reason": "model_declined" if refused else None,
        "answer": answer_text,
        "sources": source_summary,
        "citations": citations,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        generate_answer(" ".join(sys.argv[1:]))
    else:
        print("Usage: python generate.py <question text>")
