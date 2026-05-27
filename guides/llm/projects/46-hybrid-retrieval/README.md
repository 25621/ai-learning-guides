# Hybrid Retrieval

---

> Meaning and keywords each catch what the other misses.

---

## Key Insight

Dense [embedding](/shared/glossary/#embedding) search matches *meaning* while sparse keyword search ([BM25](/shared/glossary/#bm25)) matches *exact words*; [hybrid retrieval](/shared/glossary/#hybrid-retrieval) fuses their two ranked lists with [reciprocal rank fusion](/shared/glossary/#reciprocal-rank-fusion) so each covers the other's blind spots.

## Why This Matters

Embeddings miss rare names, codes, and exact phrases that keyword search nails — and vice versa — so combining them reliably beats either method alone on real-world documents.
