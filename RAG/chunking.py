"""
chunking.py — structure-aware markdown chunker, now with parent/child
(small-to-big) output. See docs/parent-child-retrieval.md for the full design
rationale — this docstring covers just the mechanism.

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
   These are the PARENT chunks — saved to chunks.jsonl, unchanged from before.
5. NEW: each parent is further re-split into smaller CHILD chunks (under
   CHILD_MAX_TOKENS), reusing the same paragraph/code-fence-aware splitter.
   Children are what actually get embedded and searched going forward — small
   and precise, so a query about one narrow detail isn't competing against
   everything else merged into a 700-token parent. Each child stores
   parent_id (the parent's position in chunks.jsonl) so a retrieval-time match
   on a child can be "promoted" back to its richer parent. Saved to
   child_chunks.jsonl.

Requires: pip install tiktoken
"""

import json
import re 
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")  # approximation of Bedrock model tokenization; good enough for sizing
MAX_TOKENS = 700          # parent (section-level) chunk budget — unchanged
CHILD_MAX_TOKENS = 150    # child (embedded/searched) chunk budget — ~1/5 of parent, for precise matching
OVERLAP_RATIO = 0.15

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)")
FENCE_RE = re.compile(r"^\s*```")


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


@dataclass
class Section:
    parent_path: list[str]   # header breadcrumb down to (not including) this section's own title(s)
    titles: list[str]        # one leaf title normally; multiple after a sibling merge
    level: int                # header depth (1 = H1, 2 = H2, ...); 0 = file has no headers at all
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
            sections.append(Section(parent_path=list(current_parent), titles=titles,
                                      level=current_level, text=body))

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
        sections.append(Section(parent_path=[], titles=[], level=0, text=md_text.strip()))

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
        merged.append(Section(parent_path=sec.parent_path, titles=list(sec.titles),
                                level=sec.level, text=sec.text))
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


def split_oversized(section: Section, max_tokens: int, overlap_ratio: float) -> list[Section]:
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

    return [Section(parent_path=section.parent_path, titles=section.titles,
                      level=section.level, text=piece) for piece in pieces]


# ---------- Step 4: assemble final chunks with breadcrumb + metadata ----------

@dataclass
class Chunk:
    text: str            # breadcrumb-prefixed — what gets embedded/displayed for this parent
    raw_text: str         # pre-breadcrumb body text — kept so children can be re-split from the
                           # same source and get the breadcrumb applied fresh to EACH child,
                           # instead of only the first one (see build_child_chunks)
    source_file: str
    header_path: str
    chunk_index: int
    token_count: int


def build_chunks_for_file(path: Path, max_tokens: int = MAX_TOKENS,
                            overlap_ratio: float = OVERLAP_RATIO) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    sections = parse_sections(raw)
    sections = merge_siblings(sections, max_tokens)

    chunks: list[Chunk] = []
    idx = 0
    for sec in sections:
        for piece in split_oversized(sec, max_tokens, overlap_ratio):
            breadcrumb = " > ".join(piece.header_path)
            full_text = f"{breadcrumb}\n\n{piece.text}" if breadcrumb else piece.text
            chunks.append(Chunk(
                text=full_text,
                raw_text=piece.text,
                source_file=path.name,
                header_path=breadcrumb,
                chunk_index=idx,
                token_count=count_tokens(full_text),
            ))
            idx += 1
    return chunks


def save_chunks(chunks: list[Chunk], out_path: Path) -> None:
    """Persist chunks as JSONL — one JSON object per line, one line per chunk.
    This file is the handoff artifact the embedding step will read from next."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "text": c.text,
                "source_file": c.source_file,
                "header_path": c.header_path,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
            }) + "\n")


# ---------- Step 5 (new): split parents into smaller children for embedding ----------

@dataclass
class ChildChunk:
    text: str
    source_file: str
    header_path: str
    parent_id: int    # position of the parent in the GLOBAL chunks list (chunks.jsonl row
                        # order) — NOT the per-file chunk_index above, which resets to 0 for
                        # every file and was never meant to be globally unique. parent_id is
                        # chosen to match the row order chunks end up in once embedded and
                        # indexed, so metadata[parent_id] is a direct O(1) lookup at query time.
    child_index: int   # this child's position within its own parent (0-based)
    token_count: int


def split_into_children(text: str, max_tokens: int = CHILD_MAX_TOKENS) -> list[str]:
    """Same paragraph/code-fence-aware grouping as split_oversized, but no
    overlap between pieces — children are precision-matching windows into a
    parent's text, not standalone context themselves (the parent, once
    promoted at retrieval time, is what actually carries full context to the
    LLM), so losing continuity between two children of the same parent loses
    nothing that isn't recovered the moment either one matches."""
    blocks = split_into_blocks(text)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for block in blocks:
        block_tokens = count_tokens(block)
        if current and current_tokens + block_tokens > max_tokens:
            pieces.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(block)
        current_tokens += block_tokens
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def build_child_chunks(parent_chunks: list[Chunk]) -> list[ChildChunk]:
    """Re-split every parent's raw (pre-breadcrumb) text into smaller children,
    prepending the SAME breadcrumb to each child individually — mirroring
    exactly how parents got their breadcrumb, just applied per-child so a
    child late in a long parent doesn't lose topic context either."""
    children: list[ChildChunk] = []
    for parent_id, parent in enumerate(parent_chunks):
        breadcrumb = parent.header_path
        for child_index, piece in enumerate(split_into_children(parent.raw_text)):
            full_text = f"{breadcrumb}\n\n{piece}" if breadcrumb else piece
            children.append(ChildChunk(
                text=full_text,
                source_file=parent.source_file,
                header_path=breadcrumb,
                parent_id=parent_id,
                child_index=child_index,
                token_count=count_tokens(full_text),
            ))
    return children


def save_child_chunks(children: list[ChildChunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in children:
            f.write(json.dumps({
                "text": c.text,
                "source_file": c.source_file,
                "header_path": c.header_path,
                "parent_id": c.parent_id,
                "child_index": c.child_index,
                "token_count": c.token_count,
            }) + "\n")


def main():
    data_dir = Path(__file__).parent / "data"
    out_path = Path(__file__).parent / "output" / "chunks.jsonl"
    child_out_path = Path(__file__).parent / "output" / "child_chunks.jsonl"
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
    print(f"Total PARENT chunks: {len(all_chunks)}")
    print(f"Parent token count — min: {min(token_counts)}, max: {max(token_counts)}, "
          f"avg: {sum(token_counts) / len(token_counts):.0f}")

    print("\n--- Parent chunk detail ---")
    for c in all_chunks:
        print(f"[{c.source_file}] ({c.token_count} tok) {c.header_path or '(no header)'}")

    save_chunks(all_chunks, out_path)
    print(f"\nSaved {len(all_chunks)} parent chunks to {out_path}")

    # --- children, built from the parents just assembled above ---
    all_children = build_child_chunks(all_chunks)
    child_token_counts = [c.token_count for c in all_children]
    print(f"\nTotal CHILD chunks: {len(all_children)} (from {len(all_chunks)} parents, "
          f"{len(all_children) / len(all_chunks):.1f} children/parent avg)")
    print(f"Child token count — min: {min(child_token_counts)}, max: {max(child_token_counts)}, "
          f"avg: {sum(child_token_counts) / len(child_token_counts):.0f}")

    print("\n--- Child chunk detail ---")
    for c in all_children:
        print(f"[{c.source_file}] parent={c.parent_id} child={c.child_index} "
              f"({c.token_count} tok) {c.header_path or '(no header)'}")

    save_child_chunks(all_children, child_out_path)
    print(f"\nSaved {len(all_children)} child chunks to {child_out_path}")


if __name__ == "__main__":
    main()
