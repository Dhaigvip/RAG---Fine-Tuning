"""
faiss_search.py — build a local FAISS index from embedded_chunks.jsonl and
search it with a query. Entirely local/free except the query embedding call
itself, which still goes to Bedrock (you need a vector for the query text
using the same model as the chunks).

Mechanism:
1. Load embedded_chunks.jsonl (chunk text + metadata + embedding, from embed.py).
2. Build a FAISS IndexFlatIP — brute-force exact inner-product search.
   Titan v2 embeddings are already normalized (unit length, via normalize=True
   in embed.py), so inner product == cosine similarity here; no extra
   normalization step needed. Flat/exact search is deliberate at this scale
   (see vector-store-strategies.md) — approximate indexes only pay off once
   brute-force search is actually slow, typically tens of thousands+ vectors.
3. Persist the index + a metadata sidecar to disk, so searching doesn't
   require re-embedding every chunk on every run.
4. Embed the query with the SAME model/settings as the chunks — imported
   directly from embed.py, so there's one source of truth for the embedding
   config rather than two copies that could drift apart.

Requires: pip install faiss-cpu

Usage:
    python faiss_search.py --build              # build the index (run after each embed.py run)
    python faiss_search.py <query text>          # search it
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np

from embed import embed_text  # reuse the exact embedding call/config used during ingestion

INDEX_PATH = Path(__file__).parent / "output" / "faiss.index"
METADATA_PATH = Path(__file__).parent / "output" / "faiss_metadata.jsonl"
EMBEDDED_CHUNKS_PATH = Path(__file__).parent / "output" / "embedded_chunks.jsonl"


def build_index():
    chunks = [json.loads(line) for line in EMBEDDED_CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        raise RuntimeError(f"No chunks found in {EMBEDDED_CHUNKS_PATH}")

    dim = len(chunks[0]["embedding"])
    vectors = np.array([c["embedding"] for c in chunks], dtype="float32")

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "text": c["text"],
                "source_file": c["source_file"],
                "header_path": c["header_path"],
                "chunk_index": c["chunk_index"],
            }) + "\n")

    print(f"Built FAISS index: {len(chunks)} vectors, {dim} dimensions")
    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


def load_index():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError("Index not built yet — run `python faiss_search.py --build` first.")
    index = faiss.read_index(str(INDEX_PATH))
    metadata = [json.loads(line) for line in METADATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return index, metadata


def search(query: str, k: int = 3):
    index, metadata = load_index()
    query_vector, _ = embed_text(query)
    query_vector = np.array([query_vector], dtype="float32")

    scores, indices = index.search(query_vector, k)

    print(f"\nQuery: {query!r}")
    print(f"Top {k} results:\n")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            continue
        chunk = metadata[idx]
        print(f"{rank}. score={score:.4f}  [{chunk['source_file']}] {chunk['header_path'] or '(no header)'}")
        preview = chunk["text"].replace("\n", " ")[:150]
        print(f"   {preview}...\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        build_index()
    elif len(sys.argv) > 1:
        search(" ".join(sys.argv[1:]))
    else:
        print("Usage:")
        print("  python faiss_search.py --build")
        print("  python faiss_search.py <query text>")
