# Right-Sizing Experiment

---

> "Don't pay for a 70B answer when an 8B one passes the same test."

---

## Key Insight

[Right-sizing](/shared/glossary/#right-sizing) means picking the smallest model that still does the job well. This project runs the same workload on a 7B, 13B, and 70B model, measures quality with a real eval and the [cost per million tokens](/shared/glossary/#cost-per-million-tokens) of each, and recommends a tier.

## Why This Matters

Teams routinely over-serve, paying for a giant model when a smaller fine-tuned one would clear the same bar at a fraction of the cost. Measuring quality and cost side by side turns model choice from a guess into a decision backed by numbers.
