# Self-Consistency Sweep

---

> Solve the problem many times, then trust the answer that keeps coming up.

---

## Key Insight

[Self-consistency](/shared/glossary/#self-consistency) samples many independent [chain-of-thought](/shared/glossary/#cot) solutions to the same problem and takes a majority vote on the final answer. This project [sweeps](/shared/glossary/#sweep) the number of samples (1, 4, 16, 64) and plots accuracy against the compute cost.

## Why This Matters

A single reasoning path can go wrong by chance. Pooling many of them cancels out random mistakes and reliably raises accuracy — letting you trade more inference compute for better answers, no retraining required.
