# RAG Evaluation Harness

A dependency-free **retrieval-augmented generation (RAG)** toolkit with built-in
**evaluation metrics** — built to demonstrate production RAG architecture and
LLM-answer quality measurement without pulling in a heavy ML stack.

## Why

RAG systems fail silently: the retriever grabs noisy context, or the model
hallucinates beyond its sources. This harness makes the pipeline **measurable**:

- **Retrieval quality** → `hit_rate`, `mrr`, `ndcg`, `context_precision`, `context_recall`
- **Answer quality** → `faithfulness` (groundedness), `answer_relevance`

## Architecture

```
documents ──▶ chunker ──▶ retriever ──▶ pipeline ──▶ metrics
              (sentence/   (BM25 /      (ingest →     (faithfulness,
               fixed)       vector /     retrieve →    relevance,
                            hybrid RRF)  evaluate)     precision/recall)
```

## Quickstart

```bash
python3 tests/test_rag_eval.py          # run the suite (stdlib unittest)

# or use it directly
PYTHONPATH=src python3 - <<'PY'
from rag_eval import RAGPipeline

pipe = RAGPipeline(retriever="hybrid", chunk_size=200)
pipe.ingest({"kb": "Vector search uses embeddings to match meaning. BM25 is lexical."})
res = pipe.evaluate(
    "how does vector search work",
    "Vector search uses embeddings to match meaning.",
    ground_truth_terms=["embeddings", "meaning"],
)
print(res.metrics)
PY
```

## Design notes

- **No external dependencies** — pure stdlib, so it runs anywhere and is trivially
  testable. Swap `_hash_embed` for `sentence-transformers` by changing one function.
- **Deterministic embeddings** use BLAKE2b hashing (not Python's salted `hash()`),
  so results are reproducible across processes and CI runs.
- **Hybrid retrieval** fuses BM25 + vector rankings with Reciprocal Rank Fusion.
- **Reference-free metrics** approximate Ragas' faithfulness/relevance offline.

## Skills demonstrated

`Python` · `RAG architecture` · `vector search & embeddings` · `BM25/hybrid retrieval`
· `LLM answer evaluation` · `chunking strategies` · `ranking metrics (MRR/nDCG)`
· `clean packaging + tests`

## License

MIT
