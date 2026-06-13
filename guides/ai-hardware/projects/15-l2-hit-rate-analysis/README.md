# L2 Hit Rate Analysis

## Key Insight

Analyzing the [L2 cache](/shared/glossary/#l2-cache) hit rate using [Nsight Compute](/shared/glossary/#nsight-compute) reveals how effectively a kernel avoids slow off-chip memory traffic. For operations like [attention](/shared/glossary/#attention) where multiple blocks read overlapping query, key, or value parameters, a high L2 cache hit rate keeps data requests on-chip. Maximizing this hit rate reduces [HBM](/shared/glossary/#hbm) bandwidth demands and significantly accelerates overall execution.
