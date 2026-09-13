# Embedding Model Strategies — Reference

## Provider-hosted (API) options

**Amazon Titan Text Embeddings v2 — chosen.** 1024/512/256 configurable output dimensions (Matryoshka-style — one model, truncate to size), native `normalize` flag, max ~8K input tokens. Already proven working in your existing Bedrock KB setup.

**Cohere Embed v3 (also on Bedrock).** Strong multilingual performance. Distinct feature worth knowing: an `input_type` parameter that embeds queries and documents *differently* (asymmetric embedding) — the model is trained knowing which side of the search it's embedding, which measurably improves retrieval versus embedding both the same way.

**OpenAI text-embedding-3-small/large.** Very popular, cheap, large native dimensionality (3072) but truncatable. Strong general benchmark performance. Requires the OpenAI API — a separate vendor/credential outside the AWS stack.

**Voyage AI.** Anthropic's recommended embedding partner; strong retrieval benchmark results, with domain-specific variants (voyage-code-2, voyage-law-2, etc.) for specialized corpora.

**Google Vertex AI text-embedding-004/005.** The GCP-native equivalent, relevant if this project ever moves off AWS.

## Open-source / self-hosted options

Runnable locally (you already have Ollama available on this machine, so this path exists without new infra):

- **BGE (BAAI/bge-large-en-v1.5, etc.)** — consistently near the top of open-model retrieval leaderboards.
- **E5 (intfloat/e5-large-v2)** — similarly strong, but requires a specific `"query: "` / `"passage: "` text prefix convention to work correctly — an easy mistake to miss.
- **GTE (general text embeddings)** — Alibaba, strong performance, similar class to BGE/E5.
- **Nomic Embed** — open training data, long context (8192 tokens).
- **all-MiniLM-L6-v2** — much smaller and faster, common lightweight default; noticeably lower retrieval quality than the large models above.

## Selection dimensions that actually matter (not just "which model")

- **Symmetric vs. asymmetric embedding**: does the model treat queries and documents the same way, or does it need different handling (Cohere's `input_type`, E5's prefix convention)? Using a model incorrectly here silently degrades retrieval without throwing an error.
- **Dimensionality vs. quality vs. storage/latency**: does the model support clean truncation (Matryoshka-style, like Titan v2 and OpenAI v3), or is it a fixed dimension with no graceful downsizing?

### Dimensions, explained
An embedding is a list of N floats; each is one learned axis of meaning (not individually interpretable, but the *geometry* of the space is what makes retrieval work — similar meaning ends up close together). More dimensions generally means more capacity for fine-grained distinctions, with diminishing returns past a point. Cost on the other side: storage (1024 floats × 4 bytes ≈ 4KB/chunk vs. 256 ≈ 1KB) and search speed (comparing vectors, and traversing an HNSW index, both scale with dimension count) — both irrelevant at this project's scale, both real at millions of vectors.

**Why Titan v2 can offer 256/512/1024 from one model — Matryoshka Representation Learning (MRL).** A model *not* trained this way produces a vector where all dimensions matter roughly equally; chopping most of them off would produce semantic garbage, since nothing was prioritized. MRL changes the training objective: loss is computed on the full vector *and* on the first 512 *and* on the first 256, simultaneously, during the same training run — so the model is forced to front-load the most important information into the earliest dimensions, nested like Russian dolls (hence the name). That's why asking for `dimensions: 256` returns a valid, slightly-less-precise embedding rather than needing a separate reduction step (like PCA fit on your own data) after the fact.

Dimension count (output vector size) is unrelated to input token limit (~8K for Titan v2) — different knobs.

**This project**: stick with 1024 — the storage/speed savings from dropping dimensions only matter at a scale (millions of vectors, tight query latency budgets) nowhere near a personal notes KB, so there's no reason to trade away the accuracy edge of the full vector.
- **Max input length per call**: Titan v2 caps around 8K tokens per embedding call — relevant if you ever wanted to embed a whole document as one summary vector (see document summary index, chunking-strategies.md #6).
- **Multilingual need**: not relevant for this project (English notes).

## Why Titan v2, for this project specifically
Already proven working through your existing Bedrock KB workflow — no new credential or vendor to integrate. Native `normalize` + configurable dimensions removes two manual steps we'd otherwise have to write ourselves. Keeps the whole stack on one platform (matches the AWS-first decision from day 1). And at the scale of a personal notes corpus, the marginal retrieval-quality gap between Titan v2 and something like Voyage or BGE is very unlikely to be the actual bottleneck — chunking and retrieval strategy (hybrid search, reranking) will matter far more than embedding model choice here, consistent with the best-practices checklist.
