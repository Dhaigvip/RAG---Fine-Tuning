# 200 AI Engineering Interview Questions - Enhanced Edition
**Production-Ready Answers with Implementation Details, Trade-offs, and References to This Repo's Actual RAG Code**

> 📌 Boxes marked **"Your Implementation"** point to real code in `RAG/*.py` in this repo,
> with your actual eval numbers — not generic examples. Fine-tuning implementation
> hasn't started yet (see note near Q81), so that section is theory-only for now.

---

## LLM FUNDAMENTALS (Q1-20)

### Q1: What is a Large Language Model (LLM)?
A neural network trained on massive text (100B-10T tokens) to predict next tokens. It's a **probability distribution** P(next_token | previous_tokens).

**Architecture:** Transformer decoder with self-attention (O(n²) complexity) + feed-forward networks (4x hidden expansion/contraction).

**Key specs:**
- Context: Claude 200K (150K words), GPT-4 128K
- Vocab: 50K-128K tokens
- Memory: 7B model = 14GB (fp16) + 8GB KV cache per 8K context
- Speed: 50-200 tok/sec depending on quantization

**What it's NOT:** Not a database (hallucination prone), not deterministic (sampling adds noise), not true reasoning (pattern matching at scale).

**Scaling law:** Loss ∝ N^(-α) where α≈0.07. Chinchilla optimal: 20 tokens per parameter.

---

### Q2: What is tokenization and why does it matter?
**Tokenizers:** BPE (GPT/Llama), SentencePiece (T5), WordPiece (BERT).

**Key metrics:** Vocab 32K-128K; English ~1.3 char/token; CJK ~1 char/token (3-5x token expansion).

**Gotchas:** Tokenizer mismatch = garbage output. `"hello"` → 1 token, `"helloo"` → 2 tokens (not reversible cleanly).

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
tokens = tokenizer.encode("What is RAG?")  # ~6 tokens
```

---

### Q3: What is temperature in LLM generation?
**Formula:** P(token) = softmax(logits / T). T→0: deterministic. T→∞: uniform random.

**Practical settings:** 0.0-0.3 (facts/SQL/math), 0.7 (default), 1.0-1.5 (creative), >2.0 (avoid — chaotic).

**Production:** A/B test; don't assume defaults. T=0.1 for deterministic tasks.

---

### Q4: What is the difference between zero-shot, few-shot prompting?
**Zero-shot:** No examples, fastest, relies on instruction tuning.
**Few-shot (3-5 examples):** Better accuracy, more tokens. Quality > quantity; plateau after 5 examples; last example most influential (recency bias).

**When:** Zero-shot for clear tasks/low latency. Few-shot for structured outputs (classification/extraction). Fine-tune at >500 examples.

---

### Q5: What is hallucination in LLMs and why does it occur?
**Root cause:** Model optimizes next-token prediction, not factuality — pattern matching, not reasoning.

**Mitigation (ranked):**
1. **RAG** (40-60% reduction) — ground in retrieved docs
2. **Fine-tuning** — modest effect
3. **Prompt engineering** ("answer only from context") — limited
4. **Verification layer** — most important for production; never trust LLM alone for critical facts

**Production reality:** You cannot eliminate hallucination. Always verify for high-stakes use cases.

---

### Q6: What is prompt engineering?
Programming the model through natural language without weight updates.

**Hierarchy:** Clear instructions → few-shot + format → Chain-of-Thought → RAG-grounded/meta-prompting.

**Template:**
```
[ROLE] You are a {role}. Goal: {goal}.
[CONSTRAINTS] Format: {format}. Tone: {tone}.
[EXAMPLES] Input: {ex} → Output: {ex_out}
[TASK] {actual_task}
```

**Workflow:** baseline → test 10-20 inputs → find failures → iterate → stop at <1% gains.

---

### Q7: What is the context window of an LLM?
**2025 models:** Claude 3: 200K tokens. GPT-4 Turbo/Llama 3.1: 128K. Mistral: 32K.

**Lost-in-middle effect:** ~20-30% accuracy loss on middle-of-context retrieval. Put key info at start/end.

**Strategy:** Use smallest context that fits. Chunk long docs. Only include top-k relevant chunks (this is exactly what RAG retrieval does — see Q41).

---

### Q8: What is an embedding and what are they used for?
Dense vector (384-4096 dims) representing semantic meaning; cosine similarity ≈ semantic similarity.

**Best models (2025):**

| Model | Dims | Quality (MTEB) | Cost |
|-------|------|---|---|
| OpenAI text-embedding-3-large | 3072 | 63.0 | $0.02/1M |
| BGE-M3 | 1024 | 59.7 | Free |
| Nomic embed-v1.5 | 768 | 58.8 | Free |

**Gotchas:** Embedding drift on model updates (version your embeddings); OOV/rare languages poorly embedded; general models ≠ optimal for your domain.

> 📌 **Your Implementation:** `RAG/embed.py` uses **Bedrock Titan Text Embeddings v2**,
> 1024-dim, `normalize=True` (unit-length, ready for cosine similarity with no client-side
> normalization step). It embeds only the **child** chunks from `chunking.py` — see Q41
> for why parents and children are split this way.

---

### Q9: What is fine-tuning and how does it differ from prompt engineering?

| Aspect | Prompt Engineering | Fine-tuning |
|--------|---|---|
| Mechanism | Modify input only | Update weights |
| Cost | Free | $100s-$1000s |
| When | <5 examples, high variance | >500 examples, consistent |

**Rule:** Start with prompting. Switch to fine-tuning after 2-3 iterations plateau.

> 📌 **Status in this repo:** No fine-tuning implementation yet — `Fine Tune/docs/ml-fundamentals-resources.md`
> is background reading only. See the note at Q81 for suggested next steps.

---

### Q10: What is RAG (Retrieval-Augmented Generation)?
```
Query → Embed → Vector DB Search → Top-k docs → Prompt → LLM Generate (grounded) → Response + sources
```

**Why it works:** Reduces hallucination 40-60%, knowledge currency without retraining, source transparency.

**Key metrics:** Recall@k (target >80%), Precision@k (>70%), MRR (>0.8), NDCG@k (>0.75).

**When to use:** ✅ domain-specific/frequently-updated knowledge, need citations. ❌ common-sense reasoning, complex multi-hop (use agentic RAG instead — Q54).

> 📌 **Your Implementation:** This entire repo's `RAG/` folder is exactly this pipeline,
> end to end — chunking.py → embed.py → faiss_search.py → hybrid_search.py → generate.py →
> api.py. See Q41 for the full architecture diagram and Q59 for your real eval numbers
> (90.9% hit rate, 0.909 MRR on an 11-question held-out set).

---

## PROMPT ENGINEERING (Q21-40) — Key Concepts

### Q21: Chain-of-Thought (CoT) prompting
"Let's think step by step" forces decomposition, reduces hallucination in reasoning. 40-50% improvement on math, 10-20% on text tasks. ~2-3x more tokens.

### Q22: Self-consistency prompting
Sample N (3-5) diverse reasoning paths at high temperature, majority vote. 10-20% improvement on complex tasks; N× inference cost.

### Q23: ReAct (Reasoning + Acting)
Interleaves Thought → Action → Observation loops with external tools (search, code exec). Better than pure CoT when external/current info needed.

> 📌 **Related to your implementation:** `RAG/query_transform.py`'s HyDE step
> (see Q47) is a simpler single-shot cousin of this idea — one LLM call to generate
> a better search input before the real retrieval happens, rather than a full
> multi-step ReAct loop.

### Q24: Prompt injection and mitigation
Layered defense: input sanitization (weak alone) → privilege separation (tag user input separately) → fine-tuning against injections → **external verification** (most important) → redundant policy checks.

### Q25-Q40: [Few-shot vs in-context learning, meta-prompting, role prompting, structured output, prompt chaining, step-back prompting, Tree-of-Thought — see full detail in prior sections of this guide]

---

## RAG & RETRIEVAL (Q41-60) — Your Core Learning Area

### Q41: How does RAG work end-to-end?

**Generic pipeline:**
```
OFFLINE: Documents → Chunk → Embed → Store in vector DB
ONLINE:  Query → Embed → ANN Search → Re-rank → Format prompt → LLM Generate
```

**Chunking strategies:** Fixed-size (simple, may split mid-sentence), semantic (topic-aware, slower), hybrid/recursive (LangChain's `RecursiveCharacterTextSplitter` — try paragraph → sentence → word boundaries in order).

**Optimal chunk size by content:** Technical docs 512-768, news 256-512, long-form 1024-2048, code 256-512.

> 📌 **Your Implementation — this is the real architecture in `RAG/`:**
> ```
> Markdown notes (data/*.md)
>     ▼
> chunking.py ──► PARENT chunks (chunks.jsonl, ≤700 tok, header-breadcrumbed,
>     │            never split mid-code-fence)
>     │            + CHILD chunks (child_chunks.jsonl, ≤150 tok, parent_id linked,
>     │            15% overlap)
>     ▼
> embed.py ──► embeds CHILDREN only (Bedrock Titan v2, 1024-dim, normalized)
>     ▼
> faiss_search.py --build ──► IndexFlatIP (exact cosine search, brute-force
>     │                       by design — see docs/vector-store-strategies.md)
>     ▼
> query_transform.py ──► HyDE: rewrite query as hypothetical answer (Q47)
>     ▼
> hybrid_search.py ──► BM25 + FAISS ──► RRF fusion ──► rerank ──► promote
>     │                child match back to its PARENT (small-to-big retrieval)
>     ▼
> generate.py ──► grounded answer, numbered citations, refuses if ungrounded (Q56)
>     ▼
> api.py ──► FastAPI: POST /ask, GET /health
> ```
> This is the **parent-child ("small-to-big") pattern**: small chunks (150 tok) are what
> get matched — precise, not diluted by unrelated content in a big chunk — but the
> **parent's** full 700-token section is what's actually shown to the LLM, so it still
> has enough surrounding context to answer well. Design rationale in full:
> `RAG/docs/parent-child-retrieval.md`.
>
> This directly answers the classic follow-up "what chunk size do you use?" — the
> honest answer is "two sizes, for two different jobs," not a single number.

---

### Q42: Dense vs sparse retrieval

**Sparse (BM25):** TF-IDF-based exact term matching. Fast, interpretable, free, misses paraphrases/synonyms.

**Dense (neural embeddings):** Cosine similarity of learned vectors. Handles paraphrases, slower, "black box," suffers embedding drift.

**Hybrid (RRF fusion):** Merge by *rank* (not raw score — scores aren't comparable scales). Typical weighting: 0.3 BM25 / 0.7 dense. Best MRR in practice (MS MARCO: BM25 0.17 → hybrid+rerank 0.40).

> 📌 **Your Implementation:** `RAG/hybrid_search.py` runs BOTH and fuses with
> Reciprocal Rank Fusion — you don't pick one. Concrete proof this matters, from your
> own measured data (`docs/reranking-strategies.md`): the same correct chunk scored
> **+4.90** for a near-exact-keyword query ("git tag") but only **+0.53** for a
> paraphrased query asking the same thing — real evidence of the exact gap this
> question is testing whether you understand.

---

### Q43: What is a vector database?
Optimized storage/search for high-dim vectors via ANN (HNSW, IVF, LSH). Pinecone (managed, easy), Weaviate (full-featured), Qdrant (fast, Rust), Milvus (scale, self-hosted).

> 📌 **Your Implementation:** You use **local FAISS** (`faiss_search.py`,
> `IndexFlatIP`) — exact brute-force search, not approximate. Deliberate choice at
> your current scale (tens of thousands of vectors is where approximate indexes
> start to pay off, not before) — see `docs/vector-store-strategies.md` for the
> reasoning. No managed vector DB cost.

---

### Q44: Chunking strategy in RAG and why it matters
Too small = fragmented context. Too large = noisy/imprecise. Wrong boundaries = split coherent ideas.

**Recursive splitting (most common in practice):** Try paragraph → sentence → word boundaries in order, preserving hierarchy.

> 📌 **Your Implementation:** `RAG/chunking.py` is structure-aware, not naive
> fixed-size — parses markdown headers into a breadcrumb path (e.g. `["System
> Design Notes", "Load Balancing", "Layer 4 vs Layer 7"]`), merges small sibling
> sections, and splits oversized ones at paragraph boundaries with an explicit
> `FENCE_RE` guard so it **never splits inside a fenced code block**. That guard is
> the difference between "read about chunking" and "hit the bug where chunking
> breaks a code block in half and ruins retrieval."

---

### Q45: What is reranking in RAG pipelines?
Cross-encoder scores query+chunk together (more accurate, slower) after fast first-stage ANN retrieval. Reduces hallucination by ensuring top chunks are truly relevant.

> 📌 **Your Implementation:** `hybrid_search.py` implements **two swappable
> backends**: `local` (sentence-transformers CrossEncoder, default, free,
> on-device) and `bedrock` (Amazon Bedrock Rerank — implemented but currently
> blocked by an AWS Organizations SCP on this account; wired up behind
> `TOOL_RAG_RERANK_BACKEND` so switching back is one env var change once access is
> granted). Also implements a **score threshold** so a query with nothing
> genuinely relevant returns *fewer* results instead of padding with
> confidently-wrong ones.

---

### Q46: Trade-offs between fine-tuning and RAG
Fine-tuning: knowledge baked into weights, fast inference, no retrieval lag, costly retraining for updates. RAG: external/updatable knowledge, retrieval latency, no retraining needed. **Hybrid (fine-tune + RAG) is the dominant production architecture** — fine-tune for style/format, RAG for facts.

---

### Q47: What is HyDE (Hypothetical Document Embeddings)?
Generate a hypothetical answer to the query, embed *that* instead of the raw query — a generated answer is phrased like the corpus documents are phrased, closing the "terse question vs. well-written answer" embedding gap.

> 📌 **Your Implementation:** `RAG/query_transform.py::generate_hyde_document()` —
> uses Claude Haiku 4.5 via Bedrock (the `eu.`-routed cross-region inference
> profile). Important subtlety from your own code comments that most textbook
> answers miss: **BM25 still searches on the raw query** — only the vector-search
> leg uses the HyDE rewrite, because HyDE fixes an embedding-similarity problem
> specifically, not a general search problem. Alternatives you considered and
> rejected (multi-query fan-out, step-back prompting, query decomposition):
> `docs/query-transformation-strategies.md`.

---

### Q48-53: Dense passage retrieval, BM25 internals, ANN algorithms (HNSW/IVF), multi-vector retrieval (ColBERT), query transformation (rewriting/expansion/decomposition), metadata filtering
[Standard concepts — see LlamaIndex/LangChain docs for canonical implementations; your repo uses query transformation via HyDE specifically, see Q47]

---

### Q54: Iterative / agentic RAG
Agent retrieves, evaluates if more retrieval needed, loops — handles multi-hop questions single-shot RAG can't. Frameworks: LangGraph, LlamaIndex Agents.

*(Not yet in this repo — your current pipeline is single-shot retrieve-then-generate. A natural next step once you want to handle multi-hop questions your eval set doesn't currently test for.)*

---

### Q55: Embedding model selection for RAG
Choose by MTEB benchmark score for your task/language. Domain-specific fine-tuning of the embedding model further boosts recall (expensive, rarely needed until you've exhausted cheaper levers like reranking and HyDE).

---

### Q56: Faithfulness vs. relevance in RAG evaluation
Faithfulness: is the answer grounded in retrieved context (no hallucination)? Relevance: are retrieved chunks actually on-topic? RAGAS framework measures both automatically in most tutorials.

> 📌 **Your Implementation — stronger than "RAGAS measures it":** `RAG/generate.py`
> **enforces** both structurally, not just measures them after the fact:
> - **Relevance gate:** if `search()` returns `[]` (nothing passed the rerank
>   threshold), it refuses *before* calling the LLM — zero wasted generation calls.
> - **Faithfulness enforcement:** prompt requires numbered citations (`[1]`, `[2]`)
>   for every claim; citations are validated post-hoc against the real source count
>   (out-of-range numbers flagged, never silently hidden); a fixed refusal phrase
>   is detected by substring match so refusal is checkable in code, not just by a
>   human reading the output.

---

### Q57: Document ingestion pipeline for RAG
Load → parse (PDF/HTML/DOCX) → clean → chunk → embed → index → store. Handle tables/images/code separately. Tools: Unstructured.io, LlamaParse, Docling.

> 📌 **Your Implementation:** `RAG/chunking.py` — see the full breakdown at Q41/Q44.
> Your corpus is markdown notes specifically, so the parser is header-aware rather
> than a general PDF/HTML/DOCX pipeline; that's the right scope for your current
> source data.

---

### Q58: Common RAG failure modes
Poor retrieval (irrelevant top-k), too much context (lost-in-middle), context contradiction (conflicting chunks), over-chunking (fragments losing meaning).

> 📌 **Your Implementation — a real, measured failure:** Your own eval set has a live
> example (`docker-02`, see Q59) — a paraphrased query missed retrieval entirely
> before HyDE. That's failure mode #1 above, caught by your own eval harness rather
> than left as an abstract textbook risk.

---

### Q59: How do you evaluate RAG systems?
Standard: Recall@k, Precision@k, MRR, NDCG@k on a held-out query set; RAGAS for automated faithfulness/relevance scoring.

> 📌 **Your Implementation — your actual numbers**, from
> `RAG/eval/results/20260914T151510Z.json` (`evaluate_retrieval.py`, 11-question set):
>
> | Metric | Overall | Exact queries | Paraphrase queries |
> |---|---|---|---|
> | Hit rate | 90.9% | 100% | 80% |
> | MRR | 0.909 | 1.0 | 0.8 |
>
> The one real miss (`docker-02` — "Why doesn't my container pick up changes after
> I edit the Dockerfile?") is a paraphrased query, exactly the failure mode Q42/Q47
> describe. **Interview-ready framing:** "I built a held-out eval set split by query
> difficulty (exact vs. paraphrase), and the exact-vs-paraphrase MRR gap is what told
> me to build HyDE" — a real diagnostic-to-fix story, not a memorized definition.

---

### Q60: Cross-encoder vs bi-encoder
Bi-encoder: encode query and doc separately, compare via cosine similarity — fast, used for first-stage retrieval (this is what `faiss_search.py` does). Cross-encoder: encode query+doc together, one score per pair — slower, more accurate, used for reranking (this is what your `local` reranker backend in `hybrid_search.py` does).

---

## TRAINING & FINE-TUNING (Q81-100) — Your Core Learning Area

> 📌 **Status in this repo:** No fine-tuning implementation exists yet —
> `Fine Tune/docs/ml-fundamentals-resources.md` is background reading only. The
> questions below are answered from first principles; once you build a fine-tuning
> pipeline, apply the same pattern used above for RAG — reference the actual LoRA
> config, dataset curation approach, and before/after eval numbers here.

### Q81: What is LoRA (Low-Rank Adaptation)?
Freeze original weights W; add small trainable matrices B (d_out×r) and A (r×d_in) where r=8-64. y = Wx + BAx. Reduces trainable params from millions to thousands.

**Memory:** Full FT 7B model ≈56GB GPU. LoRA (r=8) ≈15GB. Quality: 95-98% of full FT.

```python
from peft import get_peft_model, LoraConfig, TaskType
config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"],
                     lora_dropout=0.05, task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, config)
```

**Suggested first project for this repo:** LoRA fine-tune a small model (7B) on a
narrow task using your existing markdown notes as training signal — natural
continuation of the RAG work already done.

---

### Q82: What is PEFT (Parameter-Efficient Fine-Tuning)?
Umbrella term: LoRA (most popular, 0.05% params, 95-98% quality), QLoRA (quantized, 8GB), Adapters (2-3% params), Prompt tuning (0.01% params, weaker quality). LoRA is the industry sweet spot.

---

### Q83: What is DPO (Direct Preference Optimization)?
Re-frames RLHF as single supervised objective using (chosen, rejected) preference pairs — no reward model, no RL training needed. Formula maximizes likelihood ratio for preferred vs. dispreferred responses relative to a frozen reference model. Simpler and cheaper than PPO-based RLHF.

---

### Q84: What is catastrophic forgetting in LLM fine-tuning?
Fine-tuning overwrites pretraining knowledge. **Severity by method:** Full FT severe (20-50% general knowledge loss possible); LoRA minimal (frozen base retains knowledge). Mitigations ranked: LoRA > replay training mix (80/20) > early stopping > lower LR > EWC (rarely used, expensive).

---

### Q85: What is QLoRA?
Quantize base model to 4-bit (NF4), add LoRA adapters in fp16 on top. 7B model: 14GB → 3.7GB total. Quality: ~93-96% of full FT (4-8% drop from LoRA). Can't easily merge back into base model (quantization irreversible) — deploy as separate adapter.

---

### Q86: Pretraining pipeline of an LLM
Data collect → clean/dedup → tokenizer train → distributed training → eval. Runs weeks on thousands of GPUs.

---

### Q87: Chinchilla scaling laws *(corrected from original guide)*
Compute-optimal allocation: **tokens ≈ 20 × parameters**. For 70B params: ~1.4T tokens optimal. Key insight: most pre-2022 models (GPT-3, OPT) were undertrained relative to their size — Chinchilla and later LLaMA applied this finding and beat larger, undertrained models using the same compute budget on a smaller, better-trained model.

---

### Q91: What is SFT (Supervised Fine-Tuning)?
First alignment phase: fine-tune base LLM on curated instruction-demonstration pairs. Teaches format, dialogue structure, basic instruction following — foundation before RLHF/DPO (without good SFT, downstream preference training struggles).

---

### Q92: PPO algorithm in RLHF
Proximal Policy Optimization: optimize policy with clipped probability ratio + KL penalty against the SFT reference model (prevents over-optimization/drift). Challenges: reward hacking, mode collapse, KL calibration. DPO (Q83) is the simpler modern alternative.

---

### Q93: Adapter merging and Task Arithmetic
Task vector τ = θ_fine_tuned − θ_base. Merge multiple task vectors by addition for zero-cost multi-task deployment without joint retraining. TIES merging resolves sign conflicts between task vectors for better quality.

---

### Q94: Continual learning in LLMs
Train on sequential tasks/domains without forgetting previous ones. Key challenge: catastrophic forgetting (Q84) under standard gradient descent. Approaches: replay buffers, EWC regularization, modular adapter systems (one LoRA per task, swap at inference).

---

### Q95: Data curation for LLM fine-tuning
**Quality > quantity**: 1K high-quality examples often beats 100K noisy ones. Dedup, filter, balance — remove redundant/toxic/low-signal data. Common open datasets: Alpaca, ShareGPT, UltraChat.

**For this repo:** your markdown notes corpus (`RAG/data/*.md`) is a natural seed for a
future instruction dataset — but raw notes ≠ instruction pairs; you'd need to generate
(question, grounded-answer) pairs from them first, similar to how `eval/retrieval_eval_set.json`
was built for RAG evaluation.

---

### Q96: Role of learning rate in fine-tuning *(corrected from original guide)*
**Depends on method** — this is the key nuance the original guide's single range missed:
- Full fine-tuning: 1e-5 to 1e-4 (conservative — prevents catastrophic forgetting)
- LoRA: 1e-4 to 5e-4 (can go higher — far fewer trainable params, less disruption)
- QLoRA: similar range to LoRA

**Standard schedule:** linear warmup (~10% of steps) + cosine decay to 0. **Diagnostic:**
if training loss *increases* after the first epoch, LR is too high.

---

### Q97: Multi-task fine-tuning
Simultaneously fine-tune on multiple task datasets — improves generalization, prevents overfitting to one task. Flan-T5/FLAN collection (1800+ tasks) pioneered this at scale.

### Q98-100: Instruction hierarchy in training, preventing overfitting in fine-tuning, regularization techniques (dropout, weight decay, label smoothing)
[Standard concepts — apply once a fine-tuning pipeline exists in this repo]

---

## INFERENCE & SERVING (Q101-120) — Key Concepts

### Q101: KV caching in LLM inference
Store K/V tensors from previous tokens, avoid recomputing — reduces per-step compute from O(n) to O(1) for cached prefix positions. Memory grows linearly with sequence length × batch size.

> 📌 **Your Implementation — a real, documented gap:** `RAG/api.py` wraps
> `generate.py` in FastAPI (`POST /ask`, `GET /health`), but per
> `docs/api-strategies.md`, the **FAISS index and parent chunks are not yet cached
> across requests** — every call re-loads from disk. This is a live example for
> this exact question: you have an actual measured perf gap to point to and fix,
> not a hypothetical "what would you optimize."

### Q102: PagedAttention
Manages KV cache like OS virtual memory (fixed-size pages/blocks) — near-zero fragmentation, memory sharing across parallel sampling. Core of vLLM; 2-4× throughput improvement over naive serving.

### Q103: Continuous batching
New requests join mid-flight as others finish — eliminates idle GPU time waiting for a full batch. Combined with PagedAttention = vLLM's core innovation.

### Q104: Model quantization
FP32→INT8/INT4. GPTQ (Hessian-based error minimization, one-shot). AWQ (preserves salient weights, enables 70B on single 48GB GPU).

### Q105: TTFT and TPS metrics
TTFT (time-to-first-token): prefill-phase-dominated, the latency metric. TPS (tokens-per-second): decode-phase-dominated, the throughput metric. Disaggregating prefill/decode across GPU pools optimizes both independently.

### Q106-120: Speculative decoding, tensor/pipeline parallelism, semantic caching, model gateways, vLLM, mixed precision training, weight tying, prefill/decode phases, dynamic batching
[Standard serving-layer concepts — see original guide sections for full detail]

---

## AI AGENTS (Q61-80) — Key Concepts

### Q61: What is an AI agent?
LLM as reasoning engine in a loop: perceive → think → act → observe. Unlike single LLM calls, handles multi-step tasks with tool use. Frameworks: LangGraph, AutoGen, CrewAI.

### Q62: Model Context Protocol (MCP)
Open standard (Anthropic) for connecting AI to external tools/data sources — "USB for AI," replaces bespoke per-tool integrations with one standard interface.

### Q63-80: Tool use/function calling, multi-agent systems, agent failure modes, reactive vs. planning agents, agent memory, scratchpads, tool schemas, code execution as a tool, agents vs. workflows, LangGraph, orchestrator/sub-agent patterns
[Standard concepts — see original guide sections for full detail]

> 📌 **Related to your implementation:** Your RAG pipeline is currently a fixed
> **workflow** (Q74), not an agent — retrieve-then-generate runs the same steps every
> time. Agentic/iterative RAG (Q54) would be the natural evolution if you want the
> system itself to decide when to re-retrieve for multi-hop questions.

---

## AI AUTOMATION (Q121-140) — Key Concepts
No-code/low-code tools: n8n (self-hosted, 400+ integrations), Zapier (6000+ apps, easiest), Make (complex branching), LangChain/LlamaIndex (code-first, what this repo uses), DSPy (automatic prompt optimization), Flowise (visual LangChain builder).

---

## MACHINE LEARNING FUNDAMENTALS (Q141-160) — Key Concepts
Bias-variance trade-off, gradient descent variants (SGD/Adam), L1 vs L2 regularization, cross-validation, XGBoost, ROC-AUC vs PR-AUC, SHAP, precision/recall trade-offs, class imbalance handling, confusion matrix, ensemble methods.

> 📌 **Related to your implementation:** Your RAG eval (`evaluate_retrieval.py`)
> uses **hit rate and MRR**, information-retrieval metrics, not classification
> metrics like precision/recall/F1 — worth understanding *why* (retrieval is a
> ranking problem, not a binary classification problem) if asked to justify metric
> choice in an interview.

---

## DEEP LEARNING (Q161-180) — Key Concepts
Backpropagation, residual connections (ResNet), layer normalization, Vision Transformer, self-supervised learning, knowledge distillation, GELU activation, encoder-decoder architecture, dropout, weight initialization (Xavier/He), VAE.

---

## CAREER & PRODUCTS (Q181-200) — Key Concepts

### Q183: AI product tech stack 2025 *(corrected from original guide)*
**Serving layer, clarified:** vLLM/TGI are the actual **serving libraries**. AWS
Bedrock/Replicate are **managed LLM APIs** (no DevOps). Modal/Lambda are
**serverless compute platforms** that can *host* vLLM — Modal is not itself a
serving library, a distinction the original guide blurred.

Red-teaming, responsible AI, EU AI Act, AI observability, AI engineering career path, evaluating LLMs for production use cases, multimodal AI.

---

## Summary: What Makes This Guide Different From a Generic One

Every 📌 **Your Implementation** box above points at a real file, a real function, or a
real number from `RAG/` in this repo — not a hypothetical example. That's the difference
between reciting "RAG reduces hallucination" in an interview and saying "here's the
90.9% hit rate I measured, here's the one paraphrase query it missed, and here's the
HyDE fix I built in response." Use the boxed callouts as your interview talking points;
use the rest of the guide as the conceptual scaffolding around them.

**Fine-tuning callouts are notably sparse** — that's honest, not a gap in this guide.
Build the fine-tuning pipeline next, using the same "reference the real code and real
numbers" pattern, and this document is meant to grow with it.

---

*Last updated: September 16, 2026*
