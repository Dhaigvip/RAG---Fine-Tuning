# Chunking Strategies — Reference

The unit you choose to embed and retrieve shapes everything downstream. This is the landscape, roughly simplest to most sophisticated, with an honest read on which ones are worth using for this project.

## 1. Fixed-size / sliding window
Split every N tokens with fixed overlap, no awareness of content structure. The naive baseline — cuts sentences and ideas mid-thought. Only worth it when there's no structure to exploit at all (raw scraped text with no headers).

## 2. Recursive / structure-aware splitting — **what we built**
Split at natural boundaries (headers → paragraphs → sentences), falling to a smaller unit only when a piece is still too big. Implemented in `chunking.py`.

## 3. Semantic chunking
Embed consecutive sentences and cut where embedding similarity between neighbors drops sharply — a topic-boundary detector. More adaptive than header-based splitting, but costs an embedding call per sentence just to decide boundaries. Mainly earns its keep on unstructured prose without headers — markdown notes already have free boundaries, so lower priority here.

## 4. Parent/child (small-to-big) retrieval — **in our design**
Embed a small, precise unit; return a larger parent unit to the LLM at generation time, so matching is sharp but context is rich.

## 5. Sentence-window retrieval
A finer-grained cousin of #4: the embedded unit is a single sentence, and retrieval returns that sentence plus its N surrounding sentences (not a whole section). Useful when precision matters more than topical coherence — e.g. searching for one specific fact.

## 6. Document summary index
An LLM generates a short summary of each document/section; you embed *that summary* (denser, closer to how people phrase questions) but hand the LLM the *full original document* at retrieval time. Costs an LLM call per document during ingestion, and risks the summary omitting a detail someone later searches for. Our header-breadcrumb prepending is a free, cheap relative of this idea; a true document summary index is a heavier, more deliberate version of the same instinct.

## 7. Contextual retrieval (Anthropic's technique) — **candidate next upgrade**
Before embedding a chunk, run it through an LLM once with the full document as context and ask for a 1–2 sentence blurb situating that chunk within the document; prepend that blurb before embedding. Measured to meaningfully improve retrieval accuracy in Anthropic's own published testing. Same instinct as our header breadcrumb, but LLM-generated rather than structurally derived — the highest-leverage upgrade available given what's already built.

## 8. Proposition-based / dense-X chunking
Break text into atomic, self-contained factual statements (an LLM rewrite step resolves pronouns/context so each stands alone), then embed each separately. Very fine-grained, expensive to build (LLM call per proposition) — more common in research/high-precision QA settings than typical production RAG. Overkill for a personal notes corpus.

## 9. Late chunking
Embed the *whole* document first with a long-context embedding model to get token-level, context-aware embeddings, then pool those into chunks *after* embedding rather than splitting raw text first. Requires a long-context embedding model that exposes token-level embeddings — Titan v2 doesn't support this pattern, so it's off the table with the current stack regardless of preference.

## 10. Code-aware / AST-based chunking
Split source code at function/class boundaries using a parser, so a chunk is never half a function. Not relevant to markdown notes, but worth knowing for any future RAG work over a codebase (e.g. PALMA-adjacent).

## Verdict for this project
Keep #2 + #4 as the foundation (already built and validated). Skip #3, #6, #8, #9 — real added ingestion cost/complexity for gains that mostly matter at much larger, messier corpora than a personal notes KB. #7 (contextual retrieval) is the one worth adding as a deliberate next step, since it's a measured technique and builds directly on the breadcrumb mechanism already in place.
