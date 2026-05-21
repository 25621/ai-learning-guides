# Profile and Fix

---

> Don't guess why training is slow — measure it, then fix what the measurement shows.

---

## Key Insight

The PyTorch [profiler](/shared/glossary/#profiler) records how long each part of a training step takes, so you can see whether the [data loader](/shared/glossary/#dataloader) — not the model — is the [bottleneck](/shared/glossary/#bottleneck). Common fixes include adding [worker processes](/shared/glossary/#worker-processes) or enabling [pinned memory](/shared/glossary/#pinned-memory).

## Why This Matters

Developers routinely guess wrong about where time goes and "optimize" the wrong thing. Profiling first turns tuning from guesswork into a targeted fix — the difference between an idle GPU and a fully-fed one.
