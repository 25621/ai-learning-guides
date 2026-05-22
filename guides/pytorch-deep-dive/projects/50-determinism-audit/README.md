# Determinism Audit

---

> "It worked on my run" means nothing until the run is bit-exact every time.

---

## Key Insight

Bit-exact reproducibility means fixing every source of randomness: setting one [seed](/shared/glossary/#seed) for all the random-number generators and enabling [deterministic algorithms](/shared/glossary/#deterministic-algorithms), because some fast GPU [kernels](/shared/glossary/#kernel) give slightly different results each run by default.

## Why This Matters

Without determinism you cannot tell a real improvement from random noise, and you cannot reliably reproduce a bug. Documenting every flag needed to get identical outputs is what makes an experiment trustworthy.
