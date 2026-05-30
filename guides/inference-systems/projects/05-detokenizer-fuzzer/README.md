# Detokenizer Fuzzer

---

> `decode("a") + decode("b")` does not always equal `decode("ab")` — and a fuzzer will find the case that proves it.

---

## Key Insight

This project generates random token sequences and compares two ways of turning them back into text: piece-by-piece [detokenization](/shared/glossary/#detokenization) (one token at a time, as a streaming server does) versus all-at-once detokenization (the entire sequence in one call). Because [BPE](/shared/glossary/#bpe) merges bytes that may be invalid UTF-8 on their own, the two paths can disagree at multi-byte character boundaries — and the fuzzer hunts for examples where they do.

## A Concrete Example

The emoji 🎉 is four bytes in UTF-8: `F0 9F 8E 89`. A [BPE](/shared/glossary/#bpe) [tokenizer](/shared/glossary/#tokenizer) might split those four bytes across two tokens — say token **A** decodes to `F0 9F` and token **B** decodes to `8E 89`. Neither half is valid UTF-8 on its own.

- **Piece-by-piece (streaming):** `decode(A)` sees `F0 9F`, can't form a character, and hands back the `�` replacement glyph; `decode(B)` does the same with `8E 89`. The user sees `��`.
- **All-at-once:** `decode([A, B])` concatenates the bytes *first* — `F0 9F 8E 89` — and only then interprets them, recovering the real 🎉.

Same two tokens, two different answers. That disagreement at the multi-byte boundary is exactly the bug the fuzzer is built to surface, and it explains why a streaming server needs more machinery than a single decode call.

### Implementing all-at-once detokenization

All-at-once decoding is the simpler path to write, because you never have to deal with half a character:

1. Take the full list of token IDs.
2. For each token, look up the raw **byte string** it maps to and concatenate those bytes (not the decoded text) into one buffer.
3. Decode the whole buffer to text in a single UTF-8 pass — e.g. `byte_buffer.decode("utf-8")` in Python.

Because every multi-byte character's bytes are already sitting next to each other before decoding, valid UTF-8 always reassembles correctly. The streaming path is the hard one: it must hold back trailing bytes that *might* be the start of a multi-byte character and wait for the next token before emitting them — the same rolling-buffer idea the [stop-string matcher](/guides/inference-systems/projects/stop-string-matcher/) relies on.

## Why This Matters

Users notice when a streamed reply shows a `�` replacement glyph or a half-finished emoji, and the bug only appears on certain prompts. Production [tokenizers](/shared/glossary/#tokenizer) ship a dedicated incremental decoder that buffers partial bytes until they form valid characters — finding a disagreement by hand is the cleanest way to understand why that extra machinery exists.
