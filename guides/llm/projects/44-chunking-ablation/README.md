# Chunking Ablation

---

> The size of the pieces decides what the model gets to read.

---

## Key Insight

Before documents can be retrieved, they must be split into smaller passages — a step called [chunking](/shared/glossary/#chunking). This project is an [ablation](/shared/glossary/#ablation) that varies the chunk size (200 vs. 800 vs. 1,600 tokens) and overlap, then measures how each choice changes a [RAG](/shared/glossary/#rag) system's answer quality.

## Why This Matters

Chunk too small and a passage loses the context that makes it meaningful; chunk too large and retrieval drags in irrelevant text. Chunking is one of the quietest but highest-leverage knobs in a RAG pipeline.
