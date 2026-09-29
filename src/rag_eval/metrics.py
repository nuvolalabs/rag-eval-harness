"""RAG evaluation metrics — faithfulness, answer relevance, context precision/recall.

These are reference-free / lexical implementations of the metrics commonly
computed with Ragas. No LLM required, so they run offline and deterministically.
"""
from __future__ import annotations

import math
from collections import Counter

from .retriever import tokenize


def _terms(text: str) -> Counter:
    return Counter(tokenize(text))


def _overlap(a: Counter, b: Counter) -> float:
    """Fraction of `a`'s term mass that also appears in `b`."""
    total = sum(a.values())
    if total == 0:
        return 0.0
    shared = sum(min(a[t], b[t]) for t in a)
    return shared / total


def faithfulness(answer: str, contexts: list[str]) -> float:
    """How much of the answer is grounded in the retrieved context (0..1).

    Fraction of answer terms supported by the union of context terms.
    """
    if not answer.strip():
        return 0.0
    ctx = Counter()
    for c in contexts:
        ctx.update(_terms(c))
    return round(_overlap(_terms(answer), ctx), 4)


def answer_relevance(answer: str, query: str) -> float:
    """How on-topic the answer is for the query (0..1), query-term coverage."""
    if not query.strip():
        return 0.0
    return round(_overlap(_terms(query), _terms(answer)), 4)


def context_precision(contexts: list[str], query: str) -> float:
    """Mean relevance of retrieved contexts — how free of noise the set is (0..1)."""
    if not contexts:
        return 0.0
    q = _terms(query)
    scores = [_overlap(_terms(c), q) for c in contexts]
    return round(sum(scores) / len(scores), 4)


def context_recall(contexts: list[str], ground_truth_terms: list[str]) -> float:
    """Fraction of required ground-truth terms present somewhere in context (0..1)."""
    if not ground_truth_terms:
        return 0.0
    blob = " ".join(contexts).lower()
    hit = sum(1 for t in ground_truth_terms if t.lower() in blob)
    return round(hit / len(ground_truth_terms), 4)


def hit_rate(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Retrieval hit rate: fraction of relevant ids that were retrieved."""
    if not relevant_ids:
        return 0.0
    return round(len(set(retrieved_ids) & set(relevant_ids)) / len(set(relevant_ids)), 4)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean reciprocal rank of the first relevant retrieved id."""
    rel = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in rel:
            return round(1.0 / i, 4)
    return 0.0


def ndcg(retrieved_ids: list[str], relevant_ids: list[str], k: int | None = None) -> float:
    """Normalised discounted cumulative gain over a binary relevance list."""
    rel = set(relevant_ids)
    ranked = retrieved_ids if k is None else retrieved_ids[:k]
    dcg = sum((1.0 / math.log2(i + 2)) for i, rid in enumerate(ranked) if rid in rel)
    ideal = sum((1.0 / math.log2(i + 2)) for i in range(min(len(rel), len(ranked))))
    return round(dcg / ideal, 4) if ideal else 0.0
