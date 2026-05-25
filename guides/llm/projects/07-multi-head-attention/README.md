# Multi-Head Attention

---

> Run attention many times in parallel, each head free to focus on a different kind of relationship.

---

## Key Insight

Multi-head attention splits the model dimension into several [heads](/shared/glossary/#heads), runs [attention](/shared/glossary/#attention) in each one independently, then concatenates the results and projects them back. [Grouped-Query Attention (GQA)](/shared/glossary/#gqa) is a small twist: several query heads share one set of keys and values to shrink the [KV cache](/shared/glossary/#kv-cache).

## Why This Matters

Every production transformer uses multi-head attention, and nearly every model since 2024 uses GQA to serve faster. Verifying your version against `nn.MultiheadAttention` confirms you have the tensor reshapes exactly right — the part that is easy to get subtly wrong.
