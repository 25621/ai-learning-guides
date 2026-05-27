# Reranker Effect

---

> Cast a wide net fast, then carefully pick the best catch.

---

## Key Insight

A fast first-pass retriever is rough, so this project adds a [reranker](/shared/glossary/#reranker) — a [cross-encoder](/shared/glossary/#cross-encoder) that re-reads each top candidate together with the query and re-scores it — then measures the gain with a ranking metric like [nDCG](/shared/glossary/#ndcg) and on end-to-end answer quality.

## Why This Matters

Cheap retrieval casts a wide net, and a slow but accurate reranker then picks the true best few; this "retrieve then rerank" two-stage pattern is standard in production search and [RAG](/shared/glossary/#rag).
