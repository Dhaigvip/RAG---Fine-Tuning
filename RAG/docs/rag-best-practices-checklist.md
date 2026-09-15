# RAG Best Practices — Reference & Checklist

This doc has two halves: a **reference** (sections 1–10 below) explaining *why* each practice matters, and an **Implementation Checklist** at the bottom tracking which of them this specific project has actually done, with a short note on each item's current status. Originally drafted Sept 13 as a project-tracking doc; written into the repo itself on Sept 14 so it lives alongside the code it's tracking, not just in chat history.

## 1. Ingestion & preprocessing
**Problem**: naive text extraction throws away the document's structure (headers, sections, hierarchy) — and that structure is exactly what makes good chunking possible later. Flat, structureless text forces chunking to guess at boundaries instead of respecting real ones.
**Practice**: preserve document structure during extraction. Strip boilerplate/navigation noise but keep semantic structure. Track source, section, and position per document so every chunk can cite back to where it came from.

## 2. Chunking
**Problem**: one fixed chunk size can't serve two different jobs at once — small chunks match a query precisely but lack surrounding context; large chunks carry context but dilute the specific match a query is actually looking for.
**Practice**: fixed-size character splitting is the naive baseline and it shows — it cuts sentences and ideas in half. Recursive splitting (split on paragraph → sentence → word, in that order, only falling through when a chunk is still too big) respects natural boundaries. Starting point: 400–800 tokens/chunk, 10–15% overlap so ideas spanning a chunk boundary aren't lost. The stronger pattern once basics work: small-to-big (parent/child) retrieval — embed small, precise chunks for matching, but return the larger parent chunk to the LLM so it has full context. Attach metadata (source, section, date, doc type) to every chunk; you'll need it for filtering and citations.

## 3. Embeddings
**Problem**: embedding spaces aren't universal — vectors from two different models (or even two versions of the same model) aren't comparable, so mixing them silently corrupts similarity search.
**Practice**: pick one embedding model and never mix versions/models within an index. Normalize vectors before cosine similarity. Domain-specific embedding fine-tuning is rarely worth it for a personal KB; a strong general model (Bedrock Titan v2, Cohere embed-v3) is almost always enough — the retrieval strategy matters far more than the embedding model choice.

## 4. Vector store / indexing
**Problem**: brute-force similarity search over every vector in the corpus doesn't scale, but blindly demanding the fastest index available skips real cost/complexity tradeoffs worth understanding first.
**Practice**: approximate nearest-neighbor search (HNSW, which OpenSearch's vector engine uses) trades a small accuracy loss for massive speed over brute-force. Store metadata alongside vectors so you can pre-filter (date, doc type, tags) before similarity search — filtering reduces the search space and improves precision on constrained queries. Design the ingestion pipeline to be idempotent/rerunnable, because you will re-embed the whole corpus at least once when you change embedding models.

## 5. Retrieval
**Problem**: vector similarity alone misses exact-match cases (names, IDs, specific commands) that embeddings blur together, and a raw top-k cutoff after any ranking step can hand the LLM chunks that were never actually relevant just to fill a fixed count.
**Practice**: hybrid search (BM25 keyword + vector, merged via reciprocal rank fusion) covers both exact-match and semantic-match cases at once. Retrieve wide (top 20–50) and rerank down to what you'll actually send the LLM (top 3–5) with a cross-encoder reranker — cross-encoders score query+document jointly and are far more accurate than bi-encoder similarity, but too slow to run over a whole corpus, hence retrieve-then-rerank. A relevance-score threshold after reranking (not just a fixed top-k count) lets a query legitimately return fewer results instead of being padded with confidently-irrelevant chunks. For short or vague queries, HyDE (generate a hypothetical answer, embed *that* instead of the raw query) improves match quality by phrasing the search input the way an answer document would be phrased.

## 6. Context construction
**Problem**: an LLM doesn't read its context window uniformly — content buried in the middle gets less attention than content at the start or end ("lost in the middle") — and an unconstrained model will confidently answer from its own training data instead of the retrieved context, which is how RAG systems end up hallucinating anyway.
**Practice**: put your best-ranked chunk first or last, not buried. Explicitly instruct the model to answer only from provided context and to say when it doesn't have enough information — this is the single highest-leverage anti-hallucination move. Budget tokens deliberately: system prompt + retrieved context + question + generation headroom, not "stuff the whole window because you can."

## 7. Generation & guardrails
**Problem**: even with perfect retrieval, generation can still drift — paraphrasing itself into something the source didn't actually say, or guessing when nothing relevant was found.
**Practice**: low temperature for factual QA. For anything correctness-critical, add a citation-validation pass — check that generated claims actually appear in the retrieved chunks before returning the answer. Handle "no relevant context" as an explicit path, not a forced guess.

## 8. Evaluation (the step everyone skips, and shouldn't)
**Problem**: without a real, repeatable measurement, every change to the pipeline (chunk size, k, reranker, prompt) is judged by vibes — a manual spot-check that can't catch a regression in a query nobody happened to test that day.
**Practice**: build a real eval set — questions with known correct answers/expected source chunks, from your own corpus. Separate retrieval metrics (did we fetch the right chunks — context precision/recall, hit rate) from generation metrics (did the answer stay grounded in what was fetched — faithfulness, answer relevancy). Without that separation you can't tell if a bad answer is a retrieval failure or a generation failure. Re-run the eval every time you change chunk size, k, reranker, or prompt — otherwise "I think that helped" is a guess.

## 9. Observability
**Problem**: "why did it retrieve the wrong thing" is unanswerable after the fact if nothing was logged at the time — you're left re-running the query and guessing what changed.
**Practice**: log retrieved chunks + scores + final answer per query during development. This is how you actually debug retrieval problems instead of guessing.

## 10. Operating within real infrastructure constraints (added Sept 13)
**Problem**: a design that only works because every model/service happens to be reachable in a personal sandbox account can break outright in a governed production account — and IAM permissions looking sufficient doesn't guarantee a call will actually succeed, because governance controls above IAM (like AWS Organizations SCPs) can silently deny it.
**Practice**: production AWS accounts are often centrally governed with narrower model/service access than a personal account or sandbox — don't assume every Bedrock model, or every AWS service, is actually reachable just because IAM permissions look sufficient. Design swappable backends behind one interface where a vendor/model dependency might be restricted, rather than hard-coding to it — makes the system portable across governance environments instead of a rewrite when a policy changes.

## 11. RAGAS evaluation (added Sept 14 — tracked separately from section 8)
**Problem**: section 8's hand-rolled Hit Rate/MRR harness (`evaluate_retrieval.py`) answers "did retrieval find the right thing," using this project's own metric definitions — useful, but not the same as being able to say "we evaluated with RAGAS," the framework an interviewer or a future teammate is most likely to actually recognize by name, and it doesn't cover generation-quality metrics (faithfulness, answer relevancy) at all, by design (see `evaluation-strategies.md`'s deferral reasoning — no generated answer exists yet to check groundedness against).
**Practice**: RAGAS provides two families of metrics. Retrieval-side: Context Precision and Context Recall, each available in an LLM-judged form (compares retrieved contexts against a reference *answer*) or a non-LLM form (compares retrieved contexts against reference *contexts* directly via string similarity — no LLM call, cheapest, closest to what `evaluate_retrieval.py` already does by hand). Generation-side: Faithfulness (decomposes the generated answer into individual claims, checks each against retrieved context — no ground truth needed) and Response/Answer Relevancy (checks the answer actually addresses the question asked). This project deliberately deferred adopting RAGAS itself — its documented AWS Bedrock integration goes through `langchain-aws`'s `ChatBedrock` wrapper, which breaks this project's standing raw-`boto3`-only pattern (the same reasoning that avoided a cross-region inference profile for HyDE) — but it's tracked here as its own section specifically so that decision stays visible and gets revisited on its own timeline, not silently forgotten inside the general evaluation section. Full comparison against the hand-rolled approach, DeepEval, and TruLens is in `evaluation-strategies.md`.

## 12. TruLens evaluation (added Sept 14 — tracked separately from sections 8 and 11)
**Problem**: both section 8's hand-rolled harness and section 11's (deferred) RAGAS adoption are offline eval — they score a fixed, hand-labeled question set, run on demand, against whatever the pipeline currently does. Neither answers a different question that matters once this system is actually being used: "what is it doing on REAL traffic, right now" — which queries are coming in, which ones are scoring badly, whether quality is drifting over time. A fixed labeled set can't see any of that because it never changes.
**Practice**: TruLens is built for that different job — production observability over one-off scoring. Its core idea is "feedback functions": the same kind of metric RAGAS computes (groundedness/faithfulness, relevance, and custom ones), but wired to run continuously over live application traces rather than a static batch, with a dashboard for inspecting individual traces (what was retrieved, what was generated, per-step scores) and tracking quality trends over time. `evaluation-strategies.md`'s original assessment (Sept 14) called this correctly: TruLens has nothing to observe yet, because there's no running application producing live queries — that only starts existing once the FastAPI endpoint from the Sept 15–16 generation-layer work is up. Tracked here as its own section, separate from the two offline approaches above, specifically because it answers a different question (production observability, not offline scoring) and its adoption timing is tied to a different milestone (a live endpoint existing) rather than to the generation layer alone. Full reasoning in `evaluation-strategies.md`.

---

## Implementation Checklist

**Ingestion**
- [x] Document loaders preserve structure (headers/sections), not just raw text — `chunking.py` parses markdown headers into a `header_path` per chunk
- [ ] Per-chunk metadata captured: source, section, date, doc type — have `source_file`, `header_path`, `chunk_index`; no date/doc-type field yet (not needed for this single-domain test corpus, would matter for a mixed real corpus)
- [x] Pipeline is idempotent/rerunnable (safe to re-run on the same corpus) — re-running `chunking.py`/`embed.py`/`faiss_search.py --build` regenerates cleanly, confirmed during the Sept 14 CRLF-corruption incident (re-running fixed it with no side effects)

**Chunking**
- [x] Recursive/structure-aware splitting, not naive fixed-character splitting
- [x] Chunk size + overlap chosen deliberately (start: 400–800 tokens, 10–15% overlap) and documented — 700 tokens / 15%, see `chunking-strategies.md`
- [x] Parent/child (small-to-big) retrieval implemented — done Sept 14, verified end-to-end on real data (11/11 tests, real `"git tag"` query promoted correctly with +7.12 relevance at child granularity vs the old parent-only +4.90); see `parent-child-retrieval.md`

**Embeddings & indexing**
- [x] One embedding model/version used consistently across the whole index — Titan v2, 1024-dim
- [x] Vectors normalized before similarity search
- [x] Metadata stored alongside vectors for filtering
- [ ] Index supports approximate NN search (HNSW) at your corpus scale — using exact search (FAISS `IndexFlatIP`) deliberately at this small prototype scale; HNSW is what the real AWS vector store (OpenSearch/S3 Vectors) would use at production scale — decision still open, see `vector-store-strategies.md`

**Retrieval**
- [x] Hybrid search (keyword + vector) implemented, not vector-only
- [x] Retrieve-then-rerank pattern in place (wide retrieval → cross-encoder rerank → top-k)
- [x] Relevance-score threshold applied after reranking, not just a fixed top-k count — see `reranking-strategies.md` ("top_k: fixed count vs relevance threshold")
- [ ] Metadata filters applied where the query implies them (date, type, etc.) — not implemented, no date/type metadata captured yet
- [x] Query transformation (HyDE/expansion) considered for short/ambiguous queries — done Sept 14 (HyDE), verified against real output; on the current 5-note test corpus it measurably changed the retrieval candidate order but not yet the final winning match on the test query used — see `query-transformation-strategies.md`

**Context & generation**
- [ ] Best chunks placed at start/end of context, not buried in the middle — not yet built, this is Sept 15–16 scope
- [ ] Explicit "answer only from context, say when you don't know" instruction in the prompt — not yet built
- [ ] Token budget for context vs question vs generation is deliberate, not maxed out by default — not yet built
- [ ] Low temperature set for factual QA — not yet built
- [ ] "No relevant context" is a handled path, not a forced hallucinated answer — not yet built (retrieval's own relevance threshold already returns zero results cleanly when nothing qualifies — `hybrid_search.py`'s `search()` — but there's no generation layer yet to wire that into a "say I don't know" response)

**Evaluation**
- [x] Labeled eval set exists (real questions + expected answers/sources from your own corpus) — `eval/retrieval_eval_set.json`, 11 hand-written questions grounded in the actual 5-note test corpus, tagged `exact`/`paraphrase`
- [~] Retrieval metrics and generation metrics tracked separately — retrieval metrics (Hit Rate, MRR) implemented and ready (`evaluate_retrieval.py`); generation metrics (faithfulness, answer relevancy) correctly deferred until the generation layer exists (Sept 15–16) — see `evaluation-strategies.md` for why scoring those now would be meaningless (no generated answer to check groundedness against)
- [ ] Eval re-run after every pipeline change before calling it "better" — harness exists but hasn't been run against real output yet; not yet a standing habit in practice

**RAGAS evaluation** (see section 11 above)
- [ ] `ragas` + `langchain-aws` installed and a Bedrock judge LLM wrapped (`ChatBedrock` → `LangchainLLMWrapper`) — not started; deliberately deferred, not blocked
- [ ] Context Precision / Context Recall run via RAGAS (non-LLM form, using this project's own `retrieval_eval_set.json` as reference contexts) against the same eval set `evaluate_retrieval.py` already covers, as a cross-check against the hand-rolled Hit Rate/MRR numbers — not started
- [ ] Faithfulness / Response Relevancy run via RAGAS — blocked on the generation layer existing (Sept 15–16), same as the hand-rolled equivalents
- [ ] Explicit decision recorded on whether to keep both (hand-rolled + RAGAS) long-term, or standardize on one, once both have real numbers to compare — not started

**TruLens evaluation** (see section 12 above)
- [ ] `trulens-eval` (or current package name) installed and a Bedrock-backed feedback function wired up — not started; blocked on a live application existing, not just deferred by choice
- [ ] FastAPI endpoint from the generation layer (Sept 15–16) instrumented so TruLens actually has real traces to observe — blocked on Sept 15–16 work
- [ ] At least one feedback function (groundedness/faithfulness or relevance) running continuously over live queries, with the dashboard actually used to inspect a real trace end-to-end — not started
- [ ] Explicit decision recorded on whether TruLens's dashboard/observability adds enough value over plain logging (`hybrid_search.py`'s existing per-query print output, section 9) to justify running it, once there's real traffic to point it at — not started

**Observability**
- [x] Retrieved chunks, scores, and final answers logged per query (at least in dev) — `hybrid_search.py` prints BM25 top, vector top, RRF-fused candidates, and final reranked results with scores per query (now optional via `verbose=False` for batch eval runs, default still on)

**Operating within real infrastructure constraints**
- [x] At least one real example of designing around a governance/vendor constraint rather than hard-coding to it — Bedrock Rerank SCP block, see `reranking-strategies.md`
