# Detokenizer Fuzzer

---

> `decode("a") + decode("b")` does not always equal `decode("ab")` — and a fuzzer will find the case that proves it.

---

## Key Insight

This project generates random token sequences and compares two ways of turning them back into text: piece-by-piece [detokenization](/shared/glossary/#detokenization) (one token at a time, as a streaming server does) versus all-at-once detokenization (the entire sequence in one call). Because [BPE](/shared/glossary/#bpe) merges bytes that may be invalid UTF-8 on their own, the two paths can disagree at multi-byte character boundaries — and the fuzzer hunts for examples where they do.

## Why This Matters

Users notice when a streamed reply shows a `�` replacement glyph or a half-finished emoji, and the bug only appears on certain prompts. Production [tokenizers](/shared/glossary/#tokenizer) ship a dedicated incremental decoder that buffers partial bytes until they form valid characters — finding a disagreement by hand is the cleanest way to understand why that extra machinery exists.
