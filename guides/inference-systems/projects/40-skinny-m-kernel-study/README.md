# Skinny-M Kernel Study

---

> A tall, thin matmul leaves the Tensor Cores half-asleep — different kernels wake them up differently.

---

## Key Insight

This project takes a decode-shaped [GEMM](/shared/glossary/#gemm) (a matrix multiply with a very small batch dimension) and compares how [cuBLAS](/shared/glossary/#cublas), a [Triton](/shared/glossary/#triton) version, and [Marlin](/shared/glossary/#marlin) perform on it — reporting [TFLOPs](/shared/glossary/#tflops) and memory bandwidth.

## Why This Matters

Decode matmuls are "skinny" (tiny M), so they barely use the [Tensor Cores](/shared/glossary/#tensor-core) and become a memory-bandwidth problem instead. Seeing how different kernels handle the same skinny shape explains why production engines refuse to call one generic matmul for everything.
