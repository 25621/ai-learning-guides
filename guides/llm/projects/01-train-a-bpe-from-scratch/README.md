# Train a BPE from Scratch

---

> A tokenizer is just a list of "merge these two symbols" rules, learned from data.

---

## Key Insight

[Byte-Pair Encoding (BPE)](/shared/glossary/#bpe) builds a [vocabulary](/shared/glossary/#vocabulary) by starting from raw bytes and repeatedly merging the most frequent adjacent pair into a new symbol. The ordered list of merges it learns *is* the [tokenizer](/shared/glossary/#tokenizer).

## Why This Matters

Almost every modern LLM — GPT, Llama, Mistral — tokenizes text with a BPE variant. Building one by hand, then saving and reloading it, makes concrete how raw text turns into the integer IDs a model actually reads.
