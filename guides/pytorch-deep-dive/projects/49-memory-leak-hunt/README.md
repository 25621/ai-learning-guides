# Memory Leak Hunt

---

> Memory that only ever climbs is a reference you forgot to let go of.

---

## Key Insight

A [memory leak](/shared/glossary/#memory-leak) in PyTorch usually means you kept a reference to the [loss](/shared/glossary/#loss-function) or another piece of the [computation graph](/shared/glossary/#dynamic-computation-graph) across iterations, so the memory it holds can never be freed. Comparing [memory snapshots](/shared/glossary/#memory-snapshot) taken over many steps exposes the slow, steady climb.

## Why This Matters

A leak that adds a few megabytes per step runs fine for an hour and then crashes with an out-of-memory error. Snapshots turn that mysterious late crash into an obvious upward line you can trace back to its source.
