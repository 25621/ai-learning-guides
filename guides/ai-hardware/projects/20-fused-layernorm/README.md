# Fused LayerNorm

## Key Insight

Building a fused [normalization](/shared/glossary/#normalization) and linear projection [kernel](/shared/glossary/#kernel) in [Triton](/shared/glossary/#triton) shows the power of [kernel fusion](/shared/glossary/#kernel-fusion) in reducing memory traffic. Instead of writing intermediate LayerNorm activations back to slow global memory, we keep them in fast on-chip registers for the subsequent linear calculation. Benchmarking this fused operation against eager PyTorch execution illustrates how minimizing memory roundtrips yields significant throughput gains for memory-bound networks.
