# Interview Questions — Rehearsed Answers

Structured answers to practice out loud, not read verbatim. Update as the hands-on build progresses — especially Q2, which should get sharper as retrieval/eval work finishes.

## 1. Explain how the Transformer architecture and self-attention work.

Input tokens become embeddings, plus a positional encoding added in (since attention itself has no inherent sense of word order — it needs to be told). Self-attention: for every token, the model computes three vectors — a Query, a Key, and a Value (via learned projection matrices). A token's Query is compared (dot product) against every other token's Key to produce a raw attention score — how relevant is that other token to this one. Those scores are softmaxed into weights that sum to 1, and the token's new representation becomes a weighted sum of every token's Value vector, using those weights. The effect: each token's representation gets updated to incorporate context from every other token in the sequence *simultaneously*, unlike an RNN which has to process tokens one at a time in order.

**Multi-head attention** runs this whole process several times in parallel with different learned projections, so different heads can specialize — one might track syntactic relationships (subject-verb agreement), another semantic ones (which noun a pronoun refers to). Outputs from all heads get concatenated and projected back down. This whole block (attention + a feedforward layer, with residual connections and normalization around each) is stacked N times to form the full model depth.

**Why it matters for both RAG and fine-tuning**: embeddings (used throughout our RAG build) are themselves a byproduct of a Transformer's internal representations; fine-tuning directly adjusts the weights inside these attention/feedforward blocks (or, with LoRA, adds small trainable matrices alongside them) — you can't reason well about what fine-tuning is actually changing without this mental model.

## 2. How would you design a production RAG system?

This is the one to answer from direct build experience, not theory — walk through the actual decisions and why, in order:

- **Chunking**: structure-aware splitting (headers → paragraphs, not fixed-size), with sibling merging for small adjacent sections and parent/child retrieval so the embedded unit is small and precise but the LLM sees a richer parent chunk. Validated with a deliberate stress test, not just "it ran once."
- **Embeddings**: a normalized, configurable-dimension model (Titan v2 in this build) chosen for what's already integrated into the stack, not the highest benchmark score — at real scale, retrieval/reranking strategy matters more than embedding model choice.
- **Storage**: match the vector store to actual cost and query-pattern needs — a cheap pay-per-use option (S3 Vectors) for lower query volume, an option with native hybrid search (OpenSearch) when hybrid is a hard requirement from day one, prototyped locally (FAISS) before committing to paid infrastructure either way.
- **Retrieval**: hybrid search (BM25 keyword + vector, merged via Reciprocal Rank Fusion) rather than vector-only, because embeddings blur exact terms (names, codes, acronyms) that keyword search catches instantly, and vice versa for paraphrases.
- **Reranking**: retrieve wide, rerank down with a cross-encoder — bi-encoder similarity alone isn't precise enough for what actually goes to the LLM.
- **Generation**: explicit grounding instructions (answer only from context, say when you don't know), low temperature for factual QA, citations back to source chunks.
- **Evaluation**: a real labeled eval set from day one, retrieval metrics tracked separately from generation metrics so a bad answer can be traced to its actual cause, re-run after every pipeline change rather than trusting "it feels better."

The throughline: every one of these was a documented decision with an explicit alternative considered and a stated reason for the choice, not a default.

## 3. Fine-tuning vs RAG vs prompt engineering — when do you use each?

**Prompt engineering** first, always — cheapest, fastest, no training data, and covers a surprising amount: format changes, tone, few-shot examples, chain-of-thought instructions. If a better prompt solves it, nothing else is justified yet.

**RAG** when the model needs access to information it wasn't trained on or that changes over time — current facts, private/proprietary data, anything that needs to be traceable back to a source. RAG doesn't change *how* the model reasons, only *what it has available* to reason over.

**Fine-tuning** when the need is a change to the model's actual behavior that prompting and context can't reliably achieve — baking in a specific output format so reliably it doesn't need re-explaining every call, domain vocabulary becoming "native" rather than something to define in-context every time, or fixing a consistent reasoning error that isn't a knowledge gap (RAG can't fix a model that has the right facts in context but still reasons about them badly).

They're not mutually exclusive — a production system commonly runs all three together: a fine-tuned model (behavior), pulling from RAG (facts), governed by careful prompting (format/tone) on top of both. The mistake to name explicitly in an interview: reaching for fine-tuning to fix a knowledge gap (that's what RAG is for) or reaching for RAG to fix a behavior/reasoning problem (that's what fine-tuning is for).

## 4. How do you evaluate LLM outputs and reduce hallucinations in production?

Evaluation: separate retrieval metrics from generation metrics — a wrong answer could be a retrieval failure (right information never made it into context) or a generation failure (right information was in context, model still got it wrong), and they need different fixes. Build a real labeled eval set early, not as an afterthought; use LLM-as-judge to scale past what human review can cover, but calibrate the judge against a human-labeled sample first so you trust its scores mean what you think they mean.

Reducing hallucination: explicit grounding instructions telling the model to answer only from provided context and say when it doesn't know; a citation-validation pass that checks generated claims actually appear in the retrieved chunks before returning an answer, for anything correctness-critical; lower temperature for factual QA. The point worth making explicitly: a "hallucination" is very often actually a retrieval failure wearing a generation-failure costume — always check what was retrieved before concluding the model "made something up."

## 5. Design an LLM-powered feature at scale.

Framework to walk through out loud: clarify the actual user problem first and whether it genuinely needs an LLM at all (not every feature does). Design the data flow explicitly — what's retrieved live, what's cached, what's precomputed offline. Right-size the model per step rather than using the biggest model everywhere — query rewriting and reranking can run on much cheaper/smaller models, reserve the expensive model for the step that actually needs its quality. Design for failure from the start — fallbacks, timeouts, degraded modes — not as a follow-up patch after the first outage. Plan evaluation before launch, not after, so there's a baseline to compare against once real usage starts. Plan for cost explicitly, since token usage compounds fast at scale in ways a demo never reveals. Build in observability from day one — trying to add tracing after a production incident is much harder than having it already running.
