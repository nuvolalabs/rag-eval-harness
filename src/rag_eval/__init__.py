"""RAG evaluation harness — chunking, BM25 retrieval, and answer-grounding metrics.

Pure stdlib (no external deps) so it runs anywhere. Demonstrates the RAG
architecture + evaluation skills required by applied-GenAI/LLM engineering JDs.
"""
from .chunker import chunk_document, Chunk
from .retriever import BM25Retriever, VectorRetriever, HybridRetriever
from .metrics import faithfulness, answer_relevance, context_precision, context_recall
from .pipeline import RAGPipeline

__all__ = [
    "chunk_document", "Chunk",
    "BM25Retriever", "VectorRetriever", "HybridRetriever",
    "faithfulness", "answer_relevance", "context_precision", "context_recall",
    "RAGPipeline",
]
__version__ = "1.0.0"
