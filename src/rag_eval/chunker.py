"""Document chunking strategies for RAG ingestion."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A retrievable unit of text with provenance metadata."""

    text: str
    doc_id: str
    index: int
    start: int
    end: int
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.doc_id}:{self.index}"

    def __len__(self) -> int:
        return len(self.text)


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter: split on . ! ? followed by whitespace.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def chunk_document(
    text: str,
    doc_id: str = "doc",
    strategy: str = "sentence",
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[Chunk]:
    """Chunk a document into retrievable units.

    Strategies:
      - "sentence": pack whole sentences up to chunk_size chars (preserves meaning).
      - "fixed":    fixed-width character windows with `overlap` (simple baseline).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: list[Chunk] = []

    if strategy == "fixed":
        step = chunk_size - overlap
        for i, start in enumerate(range(0, len(text), step)):
            piece = text[start : start + chunk_size]
            if not piece.strip():
                continue
            chunks.append(Chunk(piece, doc_id, i, start, start + len(piece)))
        return chunks

    if strategy != "sentence":
        raise ValueError(f"unknown strategy: {strategy}")

    sentences = _split_sentences(text)
    buf, buf_start, cursor, idx = [], 0, 0, 0
    for s in sentences:
        s_len = len(s)
        # If adding this sentence would overflow, flush the buffer first.
        if buf and sum(len(x) for x in buf) + s_len + 1 > chunk_size:
            body = " ".join(buf)
            chunks.append(Chunk(body, doc_id, idx, buf_start, buf_start + len(body)))
            idx += 1
            # carry `overlap` chars of context into the next chunk
            tail = body[-overlap:] if overlap else ""
            buf = [tail, s] if tail else [s]
            buf_start = cursor + 1 - len(tail)
        else:
            buf.append(s)
            if len(buf) == 1:
                buf_start = cursor
        cursor += s_len + 1
    if buf:
        body = " ".join(buf)
        chunks.append(Chunk(body, doc_id, idx, buf_start, buf_start + len(body)))
    return chunks
