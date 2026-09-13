"""
chunking.py — structure-aware markdown chunker.

Mechanism, in order:
1. Parse each .md file into "sections" by walking header lines (#, ##, ###...).
   Each section knows its own header breadcrumb (e.g. ["System Design Notes",
   "Load Balancing", "Layer 4 vs Layer 7"]) and the body text under that header,
   up to the next header of any level.
2. Merge consecutive sibling sections (same parent header, same depth) into one
   chunk as long as the combined size stays under MAX_TOKENS. This avoids
   producing lots of near-empty chunks when a file has many short subsections.
3. For any section (merged or not) still over MAX_TOKENS, split it further at
   paragraph boundaries — but never in the middle of a fenced ``` code block —
   with roughly OVERLAP_RATIO of a piece's tokens repeated at the start of the
   next piece, so context isn't lost right at the cut.
4. Every resulting chunk gets its header breadcrumb prepended to the text
   before embedding, so an isolated chunk doesn't lose its topic context.

Requires: pip install tiktoken
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding(
    "cl100k_base"
)  # approximation of Bedrock model tokenization; good enough for sizing
MAX_TOKENS = 700
OVERLAP_RATIO = 0.15

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)")
FENCE_RE = re.compile(r"^\s*```")


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


@dataclass
class Section:
    parent_path: list[
        str
    ]  # header breadcrumb down to (not including) this section's own title(s)
    titles: list[str]  # one leaf title normally; multiple after a sibling merge
    level: int  # header depth (1 = H1, 2 = H2, ...); 0 = file has no headers at all
    text: str

    @property
    def header_path(self) -> list[str]:
        if not self.titles:
            return self.parent_path
        return self.parent_path + [", ".join(self.titles)]


# ---------- Step 1: parse a file into header-delimited sections ----------


def parse_sections(md_text: str) -> list[Section]:
    lines = md_text.splitlines()
    header_stack: list[str] = []
    sections: list[Section] = []
    current_lines: list[str] = []
    current_parent: list[str] = []
    current_title: str | None = None
    current_level = 0

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            titles = [current_title] if current_title else []
            sections.append(
                Section(
                    parent_path=list(current_parent),
                    titles=titles,
                    level=current_level,
                    text=body,
                )
            )

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            flush()
            current_lines = []
            level = len(m.group(1))
            title = m.group(2).strip()
            header_stack = header_stack[: level - 1]
            current_parent = list(header_stack)
            current_title = title
            current_level = level
            header_stack.append(title)
        else:
            current_lines.append(line)
    flush()

    if not sections and md_text.strip():
        # No headers anywhere in the file.
        sections.append(
            Section(parent_path=[], titles=[], level=0, text=md_text.strip())
        )

    return sections


# ---------- Step 2: merge small adjacent sibling sections ----------


def merge_siblings(sections: list[Section], max_tokens: int) -> list[Section]:
    merged: list[Section] = []
    for sec in sections:
        if merged:
            prev = merged[-1]
            same_group = prev.parent_path == sec.parent_path and prev.level == sec.level
            if same_group:
                combined_tokens = count_tokens(prev.text) + count_tokens(sec.text)
                if combined_tokens <= max_tokens:
                    prev.titles = prev.titles + sec.titles
                    prev.text = prev.text + "\n\n" + sec.text
                    continue
        merged.append(
            Section(
                parent_path=sec.parent_path,
                titles=list(sec.titles),
                level=sec.level,
                text=sec.text,
            )
        )
    return merged


# ---------- Step 3: split any still-oversized section, code-block-safe ----------


def split_into_blocks(text: str) -> list[str]:
    """Paragraph-level blocks, treating a whole fenced code block as one atomic block."""
    lines = text.split("\n")
    blocks: list[str] = []
    current: list[str] = []
    in_code = False
    for line in lines:
        if FENCE_RE.match(line):
            current.append(line)
            in_code = not in_code
            if not in_code:
                blocks.append("\n".join(current))
                current = []
            continue
        if not in_code and line.strip() == "" and current:
            blocks.append("\n".join(current))
            current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return [b for b in blocks if b.strip()]


def split_oversized(
    section: Section, max_tokens: int, overlap_ratio: float
) -> list[Section]:
    if count_tokens(section.text) <= max_tokens:
        return [section]

    blocks = split_into_blocks(section.text)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = count_tokens(block)
        if current and current_tokens + block_tokens > max_tokens:
            pieces.append("\n\n".join(current))
            overlap_budget = int(max_tokens * overlap_ratio)
            carried: list[str] = []
            running = 0
            for b in reversed(current):
                bt = count_tokens(b)
                if running + bt > overlap_budget:
                    break
                carried.insert(0, b)
                running += bt
            current, current_tokens = carried, running
        current.append(block)
        current_tokens += block_tokens

    if current:
        pieces.append("\n\n".join(current))

    return [
        Section(
            parent_path=section.parent_path,
            titles=section.titles,
            level=section.level,
            text=piece,
        )
        for piece in pieces
    ]


# ---------- Step 4: assemble final chunks with breadcrumb + metadata ----------


@dataclass
class Chunk:
    text: str
    source_file: str
    header_path: str
    chunk_index: int
    token_count: int


def build_chunks_for_file(
    path: Path, max_tokens: int = MAX_TOKENS, overlap_ratio: float = OVERLAP_RATIO
) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    sections = parse_sections(raw)
    sections = merge_siblings(sections, max_tokens)

    chunks: list[Chunk] = []
    idx = 0
    for sec in sections:
        for piece in split_oversized(sec, max_tokens, overlap_ratio):
            breadcrumb = " > ".join(piece.header_path)
            full_text = f"{breadcrumb}\n\n{piece.text}" if breadcrumb else piece.text
            chunks.append(
                Chunk(
                    text=full_text,
                    source_file=path.name,
                    header_path=breadcrumb,
                    chunk_index=idx,
                    token_count=count_tokens(full_text),
                )
            )
            idx += 1
    return chunks


def save_chunks(chunks: list[Chunk], out_path: Path) -> None:
    """Persist chunks as JSONL — one JSON object per line, one line per chunk.
    This file is the handoff artifact the embedding step will read from next."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "text": c.text,
                        "source_file": c.source_file,
                        "header_path": c.header_path,
                        "chunk_index": c.chunk_index,
                        "token_count": c.token_count,
                    }
                )
                + "\n"
            )


def main():
    data_dir = Path(__file__).parent / "data"
    out_path = Path(__file__).parent / "output" / "chunks.jsonl"
    all_chunks: list[Chunk] = []

    for md_path in sorted(data_dir.glob("*.md")):
        chunks = build_chunks_for_file(md_path)
        all_chunks.extend(chunks)
        print(f"{md_path.name}: {len(chunks)} chunk(s)")

    if not all_chunks:
        print("No chunks produced — check that data/ contains .md files.")
        return

    token_counts = [c.token_count for c in all_chunks]
    print(f"\nFiles processed: {len(list(data_dir.glob('*.md')))}")
    print(f"Total chunks: {len(all_chunks)}")
    print(
        f"Token count — min: {min(token_counts)}, max: {max(token_counts)}, "
        f"avg: {sum(token_counts) / len(token_counts):.0f}"
    )

    print("\n--- Chunk detail ---")
    for c in all_chunks:
        print(
            f"[{c.source_file}] ({c.token_count} tok) {c.header_path or '(no header)'}"
        )

    save_chunks(all_chunks, out_path)
    print(f"\nSaved {len(all_chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()
