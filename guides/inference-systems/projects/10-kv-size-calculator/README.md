# KV Size Calculator

---

> On a busy server it is the KV cache, not the model weights, that usually runs out of memory first.

---

## Key Insight

This project turns the KV-cache size formula into code and sweeps its inputs — number of key/value [heads](/shared/glossary/#heads), sequence length, and [dtype](/shared/glossary/#dtype) — to plot how much GPU memory the [KV cache](/shared/glossary/#kv-cache) eats as you serve more users at once. It makes clear why tricks like [GQA](/shared/glossary/#gqa) (sharing key/value heads) and lower-precision storage matter so much.

## Why This Matters

How many users a single GPU can serve at once is set mostly by KV-cache memory, not by the size of the weights. Being able to estimate that number on a napkin tells you whether a deployment will fit in [HBM](/shared/glossary/#hbm) — before you commit money to hardware.
