"""RAG pipeline: ingest -> chunk -> index -> retrieve -> evaluate."""
from __future__ import annotations

from dataclasses import dataclass, field

from .chunker import Chunk, chunk_document
from .metrics import (
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
    hit_rate,
    mrr,
    ndcg,
)
from .retriever import BM25Retriever, HybridRetriever, VectorRetriever

_RETRIEVERS = {
    "bm25": BM25Retriever,
    "vector": VectorRetriever,
    "hybrid": HybridRetriever,
}


@dataclass
class RetrievalResult:
    query: str
    hits: list[tuple[Chunk, float]]
    metrics: dict = field(default_factory=dict)

    @property
    def contexts(self) -> list[str]:
        return [c.text for c, _ in self.hits]


class RAGPipeline:
    """Composable retrieval-augmented-generation evaluation pipeline."""

    def __init__(self, retriever: str = "hybrid", chunk_size: int = 400, overlap: int = 50):
        if retriever not in _RETRIEVERS:
            raise ValueError(f"retriever must be one of {list(_RETRIEVERS)}")
        self.retriever_name = retriever
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[Chunk] = []
        self._retriever = None

    def ingest(self, documents: dict[str, str]) -> int:
        """Ingest {doc_id: text} documents; returns the number of chunks indexed."""
        self.chunks = []
        for doc_id, text in documents.items():
            self.chunks.extend(
                chunk_document(text, doc_id, chunk_size=self.chunk_size, overlap=self.overlap)
            )
        self._retriever = _RETRIEVERS[self.retriever_name](self.chunks)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        if self._retriever is None:
            raise RuntimeError("call ingest() before retrieve()")
        hits = self._retriever.retrieve(query, top_k)
        return RetrievalResult(query=query, hits=hits, metrics={"top_score": hits[0][1] if hits else 0.0})

    def evaluate(
        self,
        query: str,
        answer: str,
        top_k: int = 5,
        ground_truth_terms: list[str] | None = None,
        relevant_ids: list[str] | None = None,
    ) -> RetrievalResult:
        """Retrieve + score an answer against its retrieved context."""
        res = self.retrieve(query, top_k)
        res.metrics.update(
            {
                "faithfulness": faithfulness(answer, res.contexts),
                "answer_relevance": answer_relevance(answer, query),
                "context_precision": context_precision(res.contexts, query),
            }
        )
        if ground_truth_terms:
            res.metrics["context_recall"] = context_recall(res.contexts, ground_truth_terms)
        if relevant_ids:
            ids = [c.id for c, _ in res.hits]
            res.metrics.update(
                {
                    "hit_rate": hit_rate(ids, relevant_ids),
                    "mrr": mrr(ids, relevant_ids),
                    "ndcg": ndcg(ids, relevant_ids),
                }
            )
        return res
