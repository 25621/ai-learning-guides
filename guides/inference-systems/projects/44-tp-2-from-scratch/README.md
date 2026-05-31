# TP=2 from Scratch

---

> Split one model across two GPUs and prove the answer doesn't change.

---

## Key Insight

This project applies [tensor parallelism](/shared/glossary/#tensor-parallelism-tp) by hand on two GPUs — splitting a model's [attention](/shared/glossary/#attention) layer [weights](/shared/glossary/#weights) across both and combining their partial results with an [all-reduce](/shared/glossary/#allreduce) — then verifies the output exactly matches the single-GPU model.

## Why This Matters

Tensor parallelism is how a model too big for one GPU still runs, but it adds communication on every layer. Building TP=2 by hand shows both how the split works and why that cross-GPU chatter — not the math — often becomes the limit on speed.
