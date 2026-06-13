# Divergence Demo

## Key Insight

Thread [divergence](/shared/glossary/#divergence) occurs when threads in a single [warp](/shared/glossary/#warp) must execute different paths of a conditional branch. Because [GPUs](/shared/glossary/#gpu) rely on the [SIMT](/shared/glossary/#simt) model, mismatched execution paths are serialized, resulting in idle threads and significant performance degradation.
