# Ring Attention From Scratch

---

> Pass the keys around the circle until every GPU has seen the whole sequence.

---

## Key Insight

This project implements [ring attention](/shared/glossary/#ring-attention) across 4 GPUs and measures how efficiently it scales at a 64k-token context. Each GPU holds one slice of the sequence ([context parallelism](/shared/glossary/#context-parallelism)) and passes its keys and values to its neighbor around a ring, round after round, until every slice has attended to every other slice — building the full [KV cache](/shared/glossary/#kv-cache) no single GPU could hold alone.

## Why This Matters

Very long contexts (100k–1M tokens) create a KV cache far too large for one device, so the sequence must be split across many. Ring attention is the standard way to compute [attention](/shared/glossary/#attention) over that split sequence with overlapping communication, and building it by hand shows exactly where the scaling efficiency is won or lost.
