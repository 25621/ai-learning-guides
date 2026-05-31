# Stream-Overlap Audit

---

> While the GPU waits on the CPU, work that could run in parallel is sitting idle.

---

## Key Insight

This project finds a spot where the CPU and GPU take turns instead of working at the same time — for example the GPU finishes a step and then sits idle while the CPU turns the new token IDs back into text ([detokenization](/shared/glossary/#detokenization)). It then runs that CPU work *alongside* the next GPU [forward pass](/shared/glossary/#forward-pass) — using separate [CUDA streams](/shared/glossary/#cuda-stream) so the two do not block each other — and measures the speedup. ("It" here is the CPU detokenization work being moved off the critical path.)

## Why This Matters

The CPU and GPU are separate processors, so in principle they *can* run at the same time — yet by default a program issues their work in a single line and waits at each step, so the GPU sits idle while the CPU works and the CPU sits idle while the GPU works, like two workers who keep stopping to watch each other instead of both staying busy. Putting the two jobs on separate streams lets them overlap, which makes a serving stack 20–40% faster for free — and auditing for these serial gaps is one of the highest-return tuning steps you can take.
