# Loss-Spike Forensics

---

> Make the loss explode on purpose, then practice the recovery you will need at 3 a.m.

---

## Key Insight

This project deliberately triggers a [loss spike](/shared/glossary/#loss-spike) — with a too-large learning rate or a poisoned batch — then diagnoses it, fixes the cause, and resumes training from the last [checkpoint](/shared/glossary/#checkpoint). It is [forensics](/shared/glossary/#forensics) applied to a training run.

## Why This Matters

Real [pretraining](/shared/glossary/#pretraining) runs spike, and a [frontier run](/shared/glossary/#frontier-run) that cannot recover cleanly wastes days of GPU time. Rehearsing the detect-rollback-skip-resume loop on a toy model builds the muscle memory that protects expensive runs.
