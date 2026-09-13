# Retrieval Concepts — HNSW & Hybrid Search, Explained

## HNSW (Hierarchical Navigable Small World)
The algorithm nearly every vector database uses for fast similarity search at scale (OpenSearch's k-NN plugin, FAISS's `IndexHNSWFlat`, Pinecone, etc.).

**Problem it solves**: comparing a query against *every* stored vector (brute force — what our FAISS `IndexFlatIP` does) is fine at small scale but scales linearly with data size — slow once you have millions of vectors.

**Core idea**: build a graph connecting each vector to a handful of "nearby" neighbors, then *navigate* toward the query instead of scanning everything. Start anywhere, repeatedly hop to whichever connected neighbor is closer to the query, stop when no neighbor helps. You get a very good (not guaranteed perfect) answer having touched only a small fraction of all vectors.

**"Hierarchical"**: multiple stacked graph layers, like zoom levels on a map. Top layer: few nodes, long-range connections (jump far fast). Lower layers: more nodes, short-range connections (fine detail). Search starts at the sparse top layer, narrows down fast, then descends layer by layer refining — continent → country → city → street, instead of checking every address on Earth.

**Tradeoff**: approximate, not exact — the greedy hop strategy can occasionally settle for the 2nd-best match instead of the true best. In exchange, search time scales with the *log* of data size instead of the size itself.

**Key parameters** (come up when configuring HNSW in OpenSearch or FAISS): `M` (connections per node — more = more accurate, more memory), `ef_construction` (how thorough the graph-building search is), `ef_search` (how thorough the query-time search is — higher = more accurate, slower).

## BM25 and hybrid search
**BM25**: a keyword-matching ranking algorithm (the modern standard for full-text search — what OpenSearch/Elasticsearch use by default). Purely about literal words present, no semantic understanding.

**Mechanism**: scores a document higher when a query term appears more often in it (term frequency, with diminishing returns — 20 occurrences isn't 20x more relevant than 1), weighted by how rare that term is across the whole corpus (inverse document frequency — "the" barely matters, a specific term like `kubectl` matters a lot), normalized for document length (a short doc mentioning the term once is weighted more than a long doc doing the same).

**Why it matters alongside vector search**: embeddings are excellent at *meaning* (paraphrase, synonyms) but can blur exact tokens that don't carry rich standalone meaning — error codes, names, acronyms, specific commands. BM25 nails those via literal matching. Neither approach alone is reliably good at both jobs.

**Hybrid search**: run the same query through both BM25 and vector similarity, get two ranked lists, merge via **Reciprocal Rank Fusion (RRF)** — for each document, `1/(rank_in_BM25 + k) + 1/(rank_in_vector + k)` (k commonly 60), summed. Documents ranking well in *both* lists rise to the top; documents in only one list still get partial credit. RRF is used specifically because raw BM25 scores and raw cosine similarity scores live on incomparable scales — ranks are always comparable, raw scores usually aren't.

**Concrete example from our own test notes**: query `"git tag"` — strong BM25 match (exact term in `note4-many-small.md`), possibly weaker vector match (short, generic word). Query `"how do I undo my last commit"` — weak BM25 match (no shared vocabulary with the notes' actual wording), strong vector match (the concept is there even without matching words). Pure vector search (what we're using right now via FAISS) catches the second case but may under-rank the first — the exact gap hybrid search closes.
