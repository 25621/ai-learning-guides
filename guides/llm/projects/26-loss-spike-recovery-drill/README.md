# Loss-Spike Recovery Drill

---

> When the loss explodes mid-run, roll back, skip the bad batch, and keep going.

---

## Key Insight

A [loss spike](/shared/glossary/#loss-spike) is a sudden jump in the training loss, usually from an outlier batch or optimizer instability. This drill catches one by watching the [gradient](/shared/glossary/#gradients) norm, rolls back to the last [checkpoint](/shared/glossary/#checkpoint), skips the offending batch, and resumes.

## Why This Matters

On a [frontier run](/shared/glossary/#frontier-run) burning millions of GPU-hours, a spike you cannot recover from cleanly can throw away days of progress. Rehearsing the catch-and-recover loop on a small model builds the reflex before the stakes are real.
