# Contamination Probe

---

> A perfect score means nothing if the model already saw the answers.

---

## Key Insight

This project searches a model's [pretraining](/shared/glossary/#pretraining) data for exact and near-duplicate (via [MinHash](/shared/glossary/#minhash)) copies of an evaluation set's questions, measuring how much [contamination](/shared/glossary/#contamination) has leaked in.

## Why This Matters

A model that memorized the test answers looks brilliant on the [benchmark](/shared/glossary/#benchmark) and fails in the real world, so checking for contamination is the only way to tell whether a high score reflects skill or leakage.
