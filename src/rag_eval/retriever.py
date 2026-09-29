"""Retrievers: BM25 (lexical), Vector (cosine over hashed embeddings), Hybrid."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from .chunker import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on",
    "with", "as", "at", "by", "this", "that", "are", "be", "was", "were",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


class BM25Retriever:
    """Okapi BM25 lexical retriever. k1/b tunable."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(c.text) for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.docs else 0.0
        self.tf = [Counter(d) for d in self.docs]
        self.df: Counter = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.n = len(self.docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # BM25 idf with +0.5 smoothing
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query_terms: list[str], i: int) -> float:
        s = 0.0
        dl = self.doc_len[i] or 1
        for t in query_terms:
            f = self.tf[i].get(t, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
            s += self._idf(t) * (f * (self.k1 + 1)) / denom
        return s

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        qt = tokenize(query)
        scored = [(self.chunks[i], self.score(qt, i)) for i in range(self.n)]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def _hash_embed(text: str, dims: int = 256) -> list[float]:
    """Deterministic hashed bag-of-words embedding (no ML deps needed).

    A stand-in for a real embedding model: same interface (fixed-dim vector),
    so swapping in sentence-transformers later is a one-line change.
    """
    vec = [0.0] * dims
    for tok in tokenize(text):
        # stable hash (Python's built-in hash() is salted per-process)
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big") % dims
        vec[h] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorRetriever:
    """Cosine-similarity retriever over deterministic hashed embeddings."""

    def __init__(self, chunks: list[Chunk], dims: int = 256):
        self.chunks = chunks
        self.dims = dims
        self.vectors = [_hash_embed(c.text, dims) for c in chunks]

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        qv = _hash_embed(query, self.dims)
        scored = [(self.chunks[i], _cosine(qv, v)) for i, v in enumerate(self.vectors)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class HybridRetriever:
    """Reciprocal-rank-fusion hybrid of BM25 + vector retrieval."""

    def __init__(self, chunks: list[Chunk], k: int = 60):
        self.bm25 = BM25Retriever(chunks)
        self.vector = VectorRetriever(chunks)
        self.k = k

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        fused: dict[str, float] = {}
        chunk_by_id = {}
        for ranker in (self.bm25, self.vector):
            for rank, (chunk, _) in enumerate(ranker.retrieve(query, top_k * 3)):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (self.k + rank + 1)
                chunk_by_id[chunk.id] = chunk
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_by_id[cid], sc) for cid, sc in ranked[:top_k]]
