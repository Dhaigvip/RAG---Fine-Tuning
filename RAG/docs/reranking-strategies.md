# Reranking Strategies — Reference

## What reranking does, mechanically
Vector similarity (bi-encoder: embed query and document separately, compare vectors) is fast but approximate — it never lets the query and document "look at" each other directly. A cross-encoder reranker feeds the query and a candidate document into the model *together*, so it can directly compare them and produce a much more precise relevance score — but this is too slow to run over an entire corpus, hence the retrieve-wide-then-rerank pattern: get a wider candidate set cheaply (vector search, or vector + BM25 fused), then rerank only that smaller set with the expensive, precise model.

## Options

**Amazon Bedrock Rerank API — chosen.** Two models available: `amazon.rerank-v1:0` and `cohere.rerank-v3-5:0`. Both confirmed available in `eu-central-1` — the region already in use for embeddings. Same IAM/SigV4 auth as everything else in this stack, no new credential or vendor. Mechanically simple: pass the query and a list of candidate document texts, get back a relevance score and reordered ranking. Using **Cohere Rerank 3.5** as the default given its established track record in RAG reranking benchmarks; `amazon.rerank-v1:0` is a one-line model-ID swap to compare directly if worth checking later. Note: `amazon.rerank-v1:0` is *not* available in `us-east-1` (only Cohere is, there) — worth remembering if the region ever changes.

**Local cross-encoder (sentence-transformers)** — e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-base/large`. Runs entirely locally, free, no API cost, but requires downloading a model (tens of MB to ~1GB) and running inference on your own CPU/GPU. Worth using instead of Bedrock's Rerank API only if avoiding all per-call cost matters more than staying on the already-working AWS/IAM stack — at this project's query volume, that cost is negligible either way.

**Cohere Rerank via Cohere's own API directly** (not through Bedrock) — same underlying model as `cohere.rerank-v3-5:0`, but a separate vendor/credential outside the AWS stack. No reason to do this when Bedrock already offers the identical model through the same auth story already in place.

## Update (Sept 13): Bedrock Rerank blocked by org policy — switched to local cross-encoder
An **explicit deny in a Service Control Policy (SCP)** blocks `bedrock:InvokeModel` on `cohere.rerank-v3-5:0`, and (confirmed) only a small approved model list is allowed — SCPs sit above IAM and can't be bypassed even by an admin role. Rather than chase a policy change for a personal learning exercise, switched to a **local cross-encoder via `sentence-transformers`** — runs entirely on-device, no AWS call, no SCP can touch it. Using `cross-encoder/ms-marco-MiniLM-L-6-v2` as the default (small, fast, well-established baseline, ~80MB download); `BAAI/bge-reranker-base` is a stronger, larger alternative worth trying if quality matters more than speed at this scale — swap via `TOOL_RAG_RERANK_MODEL` in `.env`, same pattern as everything else in this pipeline. Embeddings (Titan v2) are unaffected — only the rerank models were blocked, so the rest of the AWS-based pipeline stays as designed.

## Both backends kept in code, switchable (added Sept 13)
`step04_hybrid_search.py` doesn't delete the Bedrock Rerank implementation — it keeps `rerank_local()` (sentence-transformers, current default) and `rerank_bedrock()` (the original Bedrock Rerank API call, unused but intact) side by side, both behind identical signatures, with a `rerank()` dispatcher choosing between them via `TOOL_RAG_RERANK_BACKEND=local|bedrock` in `.env`. Reasoning: the SCP block is an account-governance fact, not a verdict on which approach is technically better — when broader rerank-model access opens up (or this moves to a personal/unrestricted AWS account), switching back is a one-line env var change, not a rewrite. Both AWS-native and self-hosted reranking are legitimate production patterns; keeping both live in one file is itself a useful thing to be able to speak to in an interview (designing for a swappable backend rather than hard-coding around a point-in-time constraint).

## top_k: fixed count vs relevance threshold (decided Sept 13 — relevance threshold, implemented)
`rerank()` originally always returned exactly `FINAL_TOP_K` results, no matter how the candidates actually scored. Two live test queries confirmed this as a repeat pattern, not a one-off:

| Query | #1 (correct match) | #2 | #3 |
|---|---|---|---|
| `"git tag"` | +4.8959 (note4, Git Commands) | -9.7445 (note2, Load Balancing) | -10.5090 (note3, Python venv) |
| `"how do I clean up branches and stashes"` | +0.5289 (note4, Git Commands) | -11.1795 (note1, Docker Compose) | -11.2510 (note2, Load Balancing) |

Both queries: exactly one genuinely relevant chunk existed in the (small, 6-chunk) test corpus, the reranker correctly identified it with a clearly positive score, and fixed top-k still padded the result with two chunks the model itself scored as confidently irrelevant (roughly -9 to -11, vs the correct match at 0.5 to 4.9). Also notable: the same correct chunk scored +4.90 on the near-exact-keyword query but only +0.53 on the paraphrased query — the model is doing real semantic matching, not just rewarding literal overlap, but it's also less *confident* the further the query drifts from the chunk's actual wording.

**Decision: implemented the relevance-threshold strategy**, per-backend since the two backends' scores aren't on the same scale (see `step04_hybrid_search.py`'s `apply_min_score()`):
- **`LOCAL_MIN_SCORE = 0.0`** (default, via `TOOL_RAG_RERANK_MIN_SCORE_LOCAL`) — the MiniLM cross-encoder's unbounded raw logits. 0.0 cleanly separated every correct match from every incorrect one across both test queries. This is an empirical observation from 2 queries on 6 chunks, not a general law for this model class — worth re-checking once real notes (more chunks, more query variety) are in the corpus.
- **`BEDROCK_MIN_SCORE = None`** (unset, via `TOOL_RAG_RERANK_MIN_SCORE_BEDROCK`) — deliberately left uncalibrated since this backend hasn't been exercised at all (SCP-blocked). Guessing a number with zero evidence behind it would be worse than not filtering — set it once the backend is actually usable and has real queries to calibrate against.
- `min_score=None` still means "no filtering" (the old always-top-k behavior) — so the mechanism is a strict superset, not a breaking change: any backend can be reverted to fixed top-k by unsetting its env var.
- `search()` now prints however many results actually passed the bar (`"Final reranked top N (requested top_k)"`), and prints an explicit "nothing confidently relevant" message instead of a result list when zero candidates clear the threshold — silence-by-omission (just showing an empty list) would be easy to misread as "the pipeline broke," not "correctly found nothing."

Covered by unit tests in `tests/test_hybrid_search.py` (see below) — `apply_min_score()` is a pure function (plain Python data in and out, no AWS or model dependency), so its threshold/boundary/empty-result behavior is tested directly without needing a live pipeline run.

## Why Bedrock Rerank (Cohere 3.5) was the original plan
Zero new infrastructure — same `boto3` session, same `.env`-configured profile and region already working for embeddings. Confirmed available in the region already chosen. Kept the retrieval pipeline consistent: chunk → embed → retrieve (vector + BM25) → rerank, all through one AWS account and one set of credentials — the reasoning was sound, an SCP simply overrides it.
