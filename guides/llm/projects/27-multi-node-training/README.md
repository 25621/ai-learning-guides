# Multi-Node Training

---

> Sixteen GPUs across two machines, one training loop — if the network can keep up.

---

## Key Insight

This project runs [FSDP](/shared/glossary/#fsdp) across 2 [nodes](/shared/glossary/#node-distributed) of 8 GPUs each, launched with [`torchrun`](/shared/glossary/#torchrun), aiming for over 70% [MFU](/shared/glossary/#mfu). Once a job spans machines, the hard part shifts from the model to the network and the orchestration between nodes.

## Why This Matters

Real pretraining spans many machines, where slow links and straggler GPUs — not the model — decide [throughput](/shared/glossary/#throughput). Keeping utilization high across nodes is the difference between a run that finishes in days and one that drags on for weeks.
