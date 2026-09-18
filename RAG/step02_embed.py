"""
step02_embed.py — generate embeddings for chunks via Bedrock Titan Text Embeddings v2.

Mechanism:
1. Read an input .jsonl of chunks from output/ (defaults to child_chunks.jsonl —
   see the "what gets embedded" note below).
2. For each chunk, call bedrock-runtime InvokeModel with amazon.titan-embed-text-v2:0,
   requesting a 1024-dim, normalized (unit-length) vector — normalize=True means
   Bedrock hands back a vector ready for cosine similarity, no client-side
   normalization step needed.
3. Attach the returned embedding to the chunk record, alongside the token count
   Bedrock billed for that call (useful for cost awareness).
4. Persist chunk + embedding + metadata to embedded_<input-name>.jsonl — the
   artifact step03_faiss_search.py's --build step reads from next.

What gets embedded (added Sept 14, parent/child retrieval stage 2):
Per parent-child-retrieval.md's chosen design, CHILDREN are now the matching
unit — they get embedded and indexed. Parents (chunks.jsonl) stay as plain
lookup records with no embedding of their own; they're fetched by position
(parent_id) only after a child match is promoted. That's why the default
input here changed from chunks.jsonl to child_chunks.jsonl. The input file is
still a CLI argument (not hardcoded) so parents can be embedded too if ever
needed — e.g. for a future document-level routing experiment — without
touching this script again.

Requires: boto3, with your AWS profile/credentials already configured
(same profile you use for Bedrock KB).

Usage:
    python step02_embed.py                    # embeds output/child_chunks.jsonl (default)
    python step02_embed.py chunks.jsonl        # embeds output/chunks.jsonl (parents) instead
"""

import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()  # reads .env from the current working directory (RAG\ when you run `python step02_embed.py` from there)

REGION = os.environ["AWS_REGION"]          # required — no silent default, fail loudly if missing
PROFILE = os.environ.get("AWS_PROFILE")    # optional — falls back to boto3's default credential chain if unset
MODEL_ID = os.environ.get("TOOL_RAG_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
DIMENSIONS = int(os.environ.get("TOOL_RAG_EMBED_DIM", "1024"))  # Titan v2 supports 256 / 512 / 1024
MAX_RETRIES = 5

_session = boto3.Session(profile_name=PROFILE, region_name=REGION) if PROFILE else boto3.Session(region_name=REGION)
bedrock = _session.client("bedrock-runtime")


def embed_text(text: str) -> tuple[list[float], int]:
    """Call Titan v2 for one piece of text. Returns (embedding_vector, input_token_count).
    Retries with exponential backoff on throttling — Bedrock rate-limits per model,
    and a loop over many chunks will hit that limit before a handful of test chunks will."""
    body = json.dumps({
        "inputText": text,
        "dimensions": DIMENSIONS,
        "normalize": True,
    })

    for attempt in range(MAX_RETRIES):
        try:
            response = bedrock.invoke_model(
                modelId=MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read())
            return payload["embedding"], payload["inputTextTokenCount"]
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "ThrottlingException" and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  throttled, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("exhausted retries calling Bedrock")


def main():
    # Which file to embed — defaults to children (the new matching unit), but
    # stays a plain CLI arg rather than hardcoded, so parents remain embeddable
    # on demand without editing this script.
    in_name = sys.argv[1] if len(sys.argv) > 1 else "child_chunks.jsonl"
    in_path = Path(__file__).parent / "output" / in_name
    out_path = Path(__file__).parent / "output" / f"embedded_{in_name}"

    chunks = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(chunks)} chunks from {in_path}")

    total_input_tokens = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            embedding, token_count = embed_text(chunk["text"])
            total_input_tokens += token_count
            chunk["embedding"] = embedding
            chunk["embedding_model"] = MODEL_ID
            chunk["embedding_dimensions"] = DIMENSIONS
            f.write(json.dumps(chunk) + "\n")
            # Children carry child_index (+ parent_id), parents carry chunk_index —
            # fall back gracefully so this print works for either input file.
            index_label = chunk.get("child_index", chunk.get("chunk_index", "?"))
            print(f"[{i + 1}/{len(chunks)}] {chunk['source_file']} chunk {index_label} "
                  f"-> {len(embedding)}-dim vector ({token_count} input tokens)")

    print(f"\nSaved {len(chunks)} embedded chunks to {out_path}")
    print(f"Total input tokens billed: {total_input_tokens}")


if __name__ == "__main__":
    main()
