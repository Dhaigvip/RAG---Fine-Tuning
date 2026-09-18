# Production Architecture Strategy — Managed vs. Custom, End to End

*Architectural overview, Sept 18. Deliberately no implementation detail — this is about where the boundaries sit and why, not what changes in which file.*

## Problem statement
Every stage of this RAG system was built by hand: chunking, embedding, indexing, hybrid retrieval, reranking, query transformation, grounded generation, and an HTTP interface. That was the correct choice for a learning project — the mechanics only become real once you've built them and watched them fail — but it leaves an unanswered question that gets more expensive the longer it goes unasked: **which of these pieces should exist at all in a production system?**

The hand-built stages are not one category of thing. Some of them are identical in every RAG system ever built and carry no product value whatsoever — nobody was ever chosen over a competitor because their embedding pipeline was hand-rolled. Others encode specific decisions about *this* corpus, *these* query shapes, and *these* observed failure modes, and are exactly where retrieval quality is won or lost. Treating both categories the same way produces one of two bad outcomes:

1. **Ship the prototype as-is** — and inherit every gap this project already documented for itself in `vector-store-strategies.md`: no durability, no incremental updates when a document changes, metadata filtering only as a workaround, no service layer, no horizontal scaling. Months of undifferentiated work stand between here and production, none of it visible to anyone using the system.
2. **Replace everything with a managed service** — and silently discard the retrieval behaviour that was deliberately designed and, in several cases, measured: the fusion of two independent signals, the query transformation that targets a real measured scoring gap, the relevance threshold calibrated against this corpus's own scores, the two-layer refusal contract.

Both are wrong for the same reason: they apply one answer to two different questions.

## Solution
Split the system at the point where value stops being generic. **Managed infrastructure owns everything up to and including where the chunks live. Custom logic owns everything about which chunks win and what gets said about them.**

That single sentence is the architecture. The rest of this document is why the line falls there, stage by stage, and what it costs.

## Landscape — three architectural postures

**Fully managed.** One call: question in, cited answer out. The provider owns parsing, chunking, embedding, the vector store, hybrid search, reranking, generation, and citation assembly. Lowest operational burden by a wide margin, and genuinely the right answer for most teams shipping most RAG features — the honest baseline against which anything else must justify itself. What you give up is the interior: retrieval becomes a black box with a small number of configuration knobs. When retrieval is subtly wrong rather than obviously broken, you have much less to inspect and far fewer places to intervene. This project already owns a concrete instance of that failure class — the reproducible `docker-02` miss, where the correct chunk never surfaced despite every mitigation in place. Diagnosing that required seeing fusion order, individual rerank scores, and the generated hypothetical passage. A black box would have surfaced the same wrong answer with nothing to look at.

**Fully custom, hardened for production.** Total control and total inspectability, at the cost of owning an enormous amount of infrastructure that produces nothing perceptible: durability and backup, incremental index updates as source documents change, a real service layer, horizontal scaling, and document parsing for formats beyond clean text. This project's own FAISS-gaps writeup is, read from the right angle, a feature list for a product AWS already sells. Building it is a serious engineering programme justified only by requirements this project does not have — hard multi-cloud portability, an unusual scale profile, or retrieval mechanics no vendor supports at any price.

**Split at the retrieval boundary — chosen.** Managed ingestion and storage; custom retrieval and generation on top. Neither extreme's failure mode: the undifferentiated infrastructure is someone else's operational problem, and the parts that determine answer quality remain inspectable and tunable.

## The architecture, stage by stage

### Ingestion — managed
Parsing, chunking, embedding, and writing to the index. Every RAG system performs these four operations in this order, and the difficulty in all four is operational rather than intellectual: keeping the index consistent as source documents change, handling the long tail of document formats, and scaling ingestion without hand-holding.

Two things make this an easier call than it first appears. First, managed **hierarchical chunking is architecturally the same parent/child decomposition this project built by hand** — matching a small unit for retrieval precision, returning a larger unit for generation context. Adopting managed ingestion therefore does not cost the retrieval design that was chosen after deliberation; it reproduces it. That the design was independently arrived at and independently shipped by AWS is reasonable evidence it was right.

Second, **parsing is a straight gain, not a trade.** This corpus is clean markdown, so parsing has never actually been exercised here — the hardest input this pipeline has ever faced is a nested heading. Real production corpora are PDFs, scans, tables, and mixed layouts, and that is precisely where managed parsing (including generative extraction for multimodal content) earns its cost. This project has no equivalent capability and building one is not a good use of its time.

The real cost to name honestly: **chunking becomes a configuration choice among a fixed set of strategies rather than arbitrary code.** The specific behaviours built here — sibling merging, code-fence safety, the soft token budget — do not survive as-is. Whether that matters is an empirical question, not a philosophical one, and this project is unusually well placed to answer it because it has a retrieval evaluation harness. The question is not "was that code good" but "does its absence move Hit Rate and MRR."

### Storage — managed
The least contested stage. Durability, CRUD as documents change, first-class metadata filtering, a service layer, and horizontal scaling are all solved by any real vector store and all absent from a local index file. Metadata filtering in particular stops being a workaround built out of side files and becomes a native query capability — which matters more than it sounds, because per-user or per-tenant access control at retrieval time is built on exactly that primitive.

One architectural constraint follows from the custom retrieval layer above it, and it is important enough to state at the architecture level rather than leaving it to implementation: **the managed store must be one that can be queried directly, not only through a top-K retrieval endpoint.** Lexical retrieval depends on corpus-wide term statistics, which cannot be reconstructed from a result set that has already been narrowed by vector similarity. A retrieval API that only ever returns the top results is sufficient for the semantic leg and structurally insufficient for the lexical one. This rules out treating the store as an opaque endpoint, and it should be a selection criterion when the store is chosen — not a discovery made later.

### Retrieval — custom
This is the crux, and the stage where the recommendation is least obvious and most worth defending. Three concrete capability differences, not preferences:

**Fusion is a toggle, not a design.** Managed hybrid search is a mode you switch on, combining lexical and semantic matching internally on the provider's terms. The custom layer fuses two independent full-corpus signals with explicit, adjustable weighting and an inspectable intermediate ranking. That difference only matters if the fusion is actually tuned — but it is a genuine capability gap, not a stylistic one.

**Query transformation solves a different problem than the managed equivalent.** Managed query transformation addresses decomposition: splitting a multi-part question into sub-questions. This project explicitly evaluated decomposition and determined its corpus doesn't need it — the queries are single-intent lookups. The transformation this project *does* use addresses the failure it actually measured: a paraphrased question scoring far weaker against its own correct chunk than a keyword-shaped one. Those are different techniques aimed at different failure modes, and the managed offering covers the one this system doesn't have.

**Thresholds are corpus-specific.** The relevance cutoff that prevents weak matches from reaching generation was calibrated against this corpus's observed score distribution, after a real incident where fixed top-K padding fed irrelevant chunks forward. That calibration is data-dependent and belongs with the retrieval logic, not in a provider's defaults.

The architectural shape that results: **retrieval reads from the managed store rather than owning it.** It consumes two access paths — a per-query retrieval interface for the semantic leg, and direct store access for the lexical leg's corpus-wide statistics — then applies fusion, reranking, thresholding, and parent promotion exactly as designed. The logic is unchanged; only its data source moves.

### Reranking — deliberately either
Architecturally this stage is a pluggable component with a stable contract: candidates in, scored candidates out. This project already built it as a dual-backend dispatch behind one interface, which was good design for reasons that were not obvious at the time and pay off precisely here — managed or self-hosted reranking becomes a configuration decision rather than an architectural one.

Worth recording as a governance lesson rather than a technical one: the managed reranker is currently blocked on this account by an organisation-level policy that sits above IAM and cannot be worked around by permissions. That is a real and generalisable argument for keeping this stage pluggable. A managed capability you are administratively forbidden from calling is, architecturally, a capability you do not have — and the systems that survive that discovery are the ones that kept a seam there.

### Generation — custom
The grounding contract is product behaviour, not infrastructure: answer only from the provided sources, cite inline, and emit a detectable refusal when the sources don't cover the question. Managed generation offers prompt customisation, but the mechanism *around* the prompt is application logic — validating citations against the real source list, surfacing invalid ones rather than silently correcting them, and distinguishing "retrieval found nothing" from "the model declined to answer," which are different diagnoses with different fixes.

This is also the layer where verification eventually attaches. The known and documented weakness of prompted refusal is that it depends on the model choosing to comply; the structural fix is an independent faithfulness check over the generated answer and its sources. That check belongs here, above the managed boundary, and its future existence is an argument for keeping this layer owned.

### Serving — thin by design
The HTTP layer already treats retrieval and generation as opaque. That is what makes the whole split viable: the interface contract doesn't know or care whether chunks came from a local index or a managed store, so the boundary can move underneath it without touching it.

## What this architecture costs
Stated plainly, because a recommendation without its costs is marketing:

- **Vendor coupling on the ingestion side.** Parsing, chunking, embedding, and storage become provider-shaped. The retrieval and generation layers stay portable, which bounds the exposure but does not eliminate it.
- **Chunking becomes configuration.** Fine-grained chunking behaviour is traded for not maintaining a chunker. Measurable, and worth measuring before accepting.
- **A consistency surface appears.** When one process owned every stage, the index could not disagree with the pipeline's expectations. With managed ingestion, the shape of what lands in the store is determined elsewhere, and the custom retrieval layer must tolerate that rather than assume it.
- **Debugging changes character.** Ingestion problems stop being something you step through and become something you infer from the outside, or raise with a provider. This is a real reduction in diagnostic power over the stage being handed away — accepted because that stage is also the one least likely to need diagnosis.

## What's deferred
Deployment topology, observability and tracing, access control at retrieval time (genuinely architectural, and built on the metadata filtering that managed storage provides), cost modelling at realistic volume, and continuous evaluation in production. Each is a real decision; none changes the boundary this document draws, which is why they can wait.

## The open question this architecture must answer
The uncomfortable one, recorded deliberately: **the argument for keeping a custom retrieval layer is currently reasoning, not evidence.** Every capability difference named above is real, but "real difference" and "difference that improves answers on this corpus" are not the same claim, and only the second one justifies maintaining code forever.

This project is unusually well positioned to settle it rather than argue it, because it already has a retrieval evaluation harness and a labelled question set. The honest version of this recommendation therefore includes its own test: **before committing to maintain a custom retrieval layer, run the existing evaluation set against a fully managed configuration as a baseline.** If managed retrieval matches the custom pipeline's Hit Rate and MRR on this corpus, the custom layer is craftsmanship rather than engineering, and the correct architecture collapses to the fully managed posture for everything except the generation contract.

That test has not been run. Until it is, the architecture above is the well-reasoned default — not a validated conclusion, and it should not be described as one.
