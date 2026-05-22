# Multi-Node Setup

---

> One machine or fifty — the code is the same; only the wiring between them changes.

---

## Key Insight

A multi-node job spreads training across more than one [node](/shared/glossary/#node-distributed) (machine), each with its own GPUs, connected over a network. [DDP](/shared/glossary/#ddp) behaves the same across nodes as within one — the new challenge is the network setup and making every process find the others.

## Why This Matters

Real large-scale training rarely fits on a single machine. Getting a job to cross node boundaries even once teaches you the networking and launch details that every big training run depends on.
