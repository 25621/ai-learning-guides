# Debug a Hang

---

> A distributed hang is almost always one rank waiting for a call the others already made.

---

## Key Insight

Every [rank](/shared/glossary/#rank) in a distributed job must take part in each [collective operation](/shared/glossary/#collective-operation); if one rank skips it, the others wait forever and the job hangs. Setting `TORCH_NCCL_BLOCKING_WAIT=1` and `NCCL_DEBUG=INFO` makes [NCCL](/shared/glossary/#nccl) report which collective is stuck and on which rank.

## Why This Matters

Hangs are the most common and most confusing distributed failure. Learning to read these signals turns a silent freeze into a quick, fixable mismatch.
