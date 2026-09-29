"""Tests for the RAG evaluation harness (stdlib unittest — no external deps)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_eval import (  # noqa: E402
    BM25Retriever,
    HybridRetriever,
    RAGPipeline,
    VectorRetriever,
    answer_relevance,
    chunk_document,
    context_precision,
    context_recall,
    faithfulness,
)
from rag_eval.metrics import hit_rate, mrr, ndcg  # noqa: E402

DOC = (
    "Retrieval-augmented generation combines a retriever with a language model. "
    "The retriever finds relevant passages from a knowledge base. "
    "The language model then generates an answer grounded in those passages. "
    "Vector search uses embeddings to match meaning, not just keywords. "
    "BM25 is a lexical ranking function based on term frequency and inverse document frequency."
)


class TestChunking(unittest.TestCase):
    def test_sentence_chunks_cover_document(self):
        chunks = chunk_document(DOC, "d1", chunk_size=200, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.doc_id == "d1" for c in chunks))
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))

    def test_fixed_chunks_respect_size(self):
        chunks = chunk_document(DOC, "d1", strategy="fixed", chunk_size=100, overlap=10)
        self.assertTrue(all(len(c) <= 100 for c in chunks))
        self.assertGreater(len(chunks), 1)

    def test_bad_params_raise(self):
        with self.assertRaises(ValueError):
            chunk_document(DOC, chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_document(DOC, chunk_size=100, overlap=100)
        with self.assertRaises(ValueError):
            chunk_document(DOC, strategy="wat")


class TestRetrievers(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_document(DOC, "d1", chunk_size=200, overlap=0)

    def test_bm25_finds_relevant_passage(self):
        hits = BM25Retriever(self.chunks).retrieve("vector search embeddings", top_k=3)
        self.assertTrue(hits)
        self.assertIn("vector", hits[0][0].text.lower())

    def test_vector_retriever_returns_ranked(self):
        hits = VectorRetriever(self.chunks).retrieve("BM25 lexical ranking", top_k=3)
        self.assertTrue(hits)
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_hybrid_retriever(self):
        hits = HybridRetriever(self.chunks).retrieve("language model answer", top_k=3)
        self.assertTrue(hits)

    def test_deterministic_embeddings_across_calls(self):
        a = VectorRetriever(self.chunks).retrieve("vector search", top_k=2)
        b = VectorRetriever(self.chunks).retrieve("vector search", top_k=2)
        self.assertEqual([c.id for c, _ in a], [c.id for c, _ in b])


class TestMetrics(unittest.TestCase):
    def test_faithfulness_grounded(self):
        ctx = ["The retriever finds relevant passages from a knowledge base."]
        ans = "The retriever finds relevant passages."
        self.assertGreater(faithfulness(ans, ctx), 0.5)

    def test_faithfulness_ungrounded(self):
        ctx = ["Apples are red."]
        self.assertLess(faithfulness("Quantum tunnelling involves wavefunctions.", ctx), 0.3)

    def test_answer_relevance(self):
        self.assertGreater(answer_relevance("vector search embeddings", "how does vector search work"), 0.3)

    def test_context_precision_and_recall(self):
        ctxs = ["vector search uses embeddings", "bananas are yellow"]
        self.assertGreater(context_precision(ctxs, "vector search embeddings"), 0.0)
        self.assertEqual(context_recall(ctxs, ["vector", "bananas"]), 1.0)

    def test_ranking_metrics(self):
        self.assertEqual(hit_rate(["a", "b", "c"], ["b"]), 1.0)
        self.assertEqual(mrr(["x", "b", "c"], ["b"]), 0.5)
        self.assertGreater(ndcg(["b", "a"], ["b"]), 0.9)


class TestPipeline(unittest.TestCase):
    def test_end_to_end(self):
        pipe = RAGPipeline(retriever="hybrid", chunk_size=200)
        n = pipe.ingest({"d1": DOC})
        self.assertGreater(n, 1)
        res = pipe.evaluate(
            "how does vector search work",
            "Vector search uses embeddings to match meaning.",
            ground_truth_terms=["embeddings", "meaning"],
            relevant_ids=None,
        )
        for key in ("faithfulness", "answer_relevance", "context_precision"):
            self.assertIn(key, res.metrics)
            self.assertGreaterEqual(res.metrics[key], 0.0)
            self.assertLessEqual(res.metrics[key], 1.0)

    def test_retrieve_before_ingest_raises(self):
        with self.assertRaises(RuntimeError):
            RAGPipeline().retrieve("x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
