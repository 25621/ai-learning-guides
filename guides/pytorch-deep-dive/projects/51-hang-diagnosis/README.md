# Hang Diagnosis

---

> A frozen distributed job is almost always one rank waiting on a call the others already made.

---

## Key Insight

In [DDP](/shared/glossary/#ddp), every [rank](/shared/glossary/#rank) must reach the same [collective operation](/shared/glossary/#collective-operation) in the same order; if one rank skips it or passes a different shape, the rest wait forever. Setting `NCCL_DEBUG=INFO` makes [NCCL](/shared/glossary/#nccl) print which collective each rank is stuck on.

## Why This Matters

A hang produces no error and no stack trace, so it is one of the hardest distributed bugs to face blind. Knowing it is almost always a mismatched collective — and how to ask NCCL where each rank stopped — turns an indefinite freeze into a quick fix.
