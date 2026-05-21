# Mini FlashAttention

---

> Never write the giant score matrix to memory — stream it through in tiles instead.

---

## Key Insight

Standard [attention](/shared/glossary/#attention) builds a large `T×T` [softmax](/shared/glossary/#softmax) score matrix in slow GPU memory ([HBM](/shared/glossary/#hbm)). [FlashAttention](/shared/glossary/#flashattention) avoids this with [tiling](/shared/glossary/#tiling) and an [online softmax](/shared/glossary/#online-softmax) that updates the running result block by block, so the full matrix is never stored. Building a small version and checking it matches eager attention shows how the same math can cost far less memory.

## Why This Matters

The single idea — do the same [FLOPs](/shared/glossary/#flops) but touch memory far less — is why FlashAttention made long-context [transformers](/shared/glossary/#transformer) practical, and it is the canonical example of memory-aware kernel design.
