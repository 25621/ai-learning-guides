# Determinism Audit

---

> Same prompt, same model, same seed — and yet the bytes can still differ on a GPU.

---

## Key Insight

This project runs the same prompt through [greedy decoding](/shared/glossary/#greedy-decoding) one hundred times at different batch sizes and records how often the outputs disagree bit-for-bit. GPU reductions (the sums inside [softmax](/shared/glossary/#softmax) and [matmul](/shared/glossary/#matmul)) accumulate floating-point values in a non-deterministic order when the batch shape changes, so a tiny rounding difference at one layer can flip the [argmax](/shared/glossary/#logits) at the next — turning "same input" into a different sentence downstream.

## Why This Matters

Greedy decoding *feels* deterministic but isn't, which catches teams by surprise during evaluation and legal review. Either spending the throughput cost to enable [deterministic algorithms](/shared/glossary/#deterministic-algorithms) or accepting the non-determinism with eyes open is a decision that should be made up front, not discovered as a bug report three months later.
