# HBM Saturation

## Key Insight

Saturating the off-chip [HBM](/shared/glossary/#hbm) bandwidth is the defining limit for memory-bound kernels on a [GPU](/shared/glossary/#gpu). In simple operations like vector addition, the low [arithmetic intensity](/shared/glossary/#ai-arithmetic-intensity) keeps the compute units idle while the hardware waits for data. Maximizing the achieved [memory bandwidth](/shared/glossary/#memory-bandwidth) on the [roofline](/shared/glossary/#roofline) diagram requires tuning instruction-level parallelism and load sizes to keep the memory controllers fully utilized.
