# Roofline by Hand

## Key Insight

Applying the [roofline](/shared/glossary/#roofline) model by hand clarifies whether an operation is bounded by [memory bandwidth](/shared/glossary/#memory-bandwidth) or arithmetic throughput. By computing the [arithmetic intensity](/shared/glossary/#ai-arithmetic-intensity) of key steps like [matmul](/shared/glossary/#matmul), [normalization](/shared/glossary/#normalization), [softmax](/shared/glossary/#softmax), and [GELU](/shared/glossary/#gelu), we can predict which kernels will run slowly on a specific [GPU](/shared/glossary/#gpu) (like an A100). This distinction prevents developers from wasting time optimizing [FLOPS](/shared/glossary/#flops) on operations that are actually bottlenecked by memory transfer.
