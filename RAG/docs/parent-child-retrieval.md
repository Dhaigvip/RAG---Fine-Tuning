# Parent/Child (Small-to-Big) Retrieval — Design

## Problem statement
Chunk size is a tug-of-war between two different jobs, and one size can't win both:
- **Matching precision wants chunks SMALL.** A chunk's embedding is one vector representing everything inside it. Our current chunks (`chunking.py`'s output, up to 700 tokens after sibling-merging) can hold three merged subsections in one chunk — so that chunk's embedding is a blend of all three topics. A query about just one of those subsections has to compete against noise from the other two baked into the same vector, and the match gets diluted. The bigger the chunk, the worse this gets.
- **Generation quality wants chunks LARGE.** Shrinking chunks to fix matching creates the opposite problem: a small, precise fragment handed to the LLM in isolation often lacks the surrounding context needed to actually answer the question — the LLM sees the trees but not the forest.

These pull in opposite directions, so no single chunk size is correct for both retrieval and generation at once. That's the actual problem — not "what's the right chunk size" (there isn't one), but "why are we forcing one unit to do two different jobs."

## Solution
Stop using one chunk for both jobs. Split the two roles: search over something SMALL and precise (the **child**) so matching stays sharp, but once a child matches, hand the LLM something LARGER and richer (its **parent**) instead of the tiny fragment itself — so generation still gets full context. This is exactly what "parent/child" or "small-to-big" retrieval names: decouple *what gets matched* from *what gets shown*.

## Design axes (what's actually configurable)

**Child granularity** — how small the searched unit is:
- *Sentence-level*: smallest possible, most precise matching, but very fine-grained — many children per parent, higher embedding-call count and index size for a personal-notes corpus this small.
- *Paragraph/block-level* — **chosen**: reuses `chunking.py`'s existing `split_into_blocks()` (already paragraph- and code-fence-aware), grouped up to a small token budget. Natural unit size for markdown notes; no new splitting logic needed, just a smaller budget applied to the same function.
- *Fixed small token window (e.g. 100 tokens, no structural awareness)*: simplest to reason about, but reintroduces the exact "cuts mid-thought" problem structure-aware splitting was built to avoid — rejected for the same reason fixed-size chunking was rejected originally.

**Parent granularity** — what gets returned once a child matches:
- *The existing section-level chunk* — **chosen**: `chunks.jsonl` as already built (sibling-merged, up to 700 tokens) becomes the parent, unchanged. Zero rework of the existing pipeline; parents are just what we already have.
- *Whole source document*: richer still, but for a personal-notes file that's often the whole note — loses the point of chunking at all, and risks blowing the context budget once a few parents are combined.

**Parent linkage** — how a child finds its parent at retrieval time:
- *Store a parent ID on each child, looked up against a separate parent list at query time* — **chosen**. The alternative (duplicating full parent text inside every child record) wastes storage and risks the two copies drifting if either changes.

## Chosen design
- Parents = existing `chunks.jsonl` (unchanged, still section-level, sibling-merged, breadcrumb-prefixed).
- Children = each parent's text re-split at paragraph/code-fence boundaries into pieces under `CHILD_MAX_TOKENS` (~150 tokens — roughly 1/5 of the parent budget), each carrying its own copy of the parent's breadcrumb (same reasoning as parents: an isolated child shouldn't lose its topic context either).
- Each child stores `parent_id` = the parent's **position in the `chunks.jsonl` list** (0-indexed) — not the existing per-file `chunk_index` field, which resets to 0 for every file and was never meant to be globally unique. `parent_id` is chosen to exactly match the row order chunks end up in once embedded and indexed (embed.py and faiss_search.py both preserve list order), so `metadata[parent_id]` at retrieval time is a direct, O(1) lookup — no separate ID-matching logic needed.
- **Embedding target changes**: children get embedded and indexed (that's what search matches against now), not parents directly. Parents stay as plain lookup records.
- **Retrieval-time promotion**: hybrid search + rerank operate on children as before, but the final step maps each surviving child back to its `parent_id`, deduplicates (multiple matching children can share one parent), and returns parent text as the actual context — this is the "small-to-big" swap.

## What's deferred
Sentence-level children and cross-file/whole-document parents are both real, documented alternatives (see above) — not implemented because the current corpus and matching failures observed so far don't call for them. Revisit sentence-level children specifically if paragraph-level children turn out too coarse once real notes (not the manufactured test set) are in the corpus.
