# Triton Matmul

---

> Matrix multiply is where the GPU lives or dies — tile it well and you can rival the vendor.

---

## Key Insight

A fast [matmul](/shared/glossary/#matmul) [kernel](/shared/glossary/#kernel) works by [tiling](/shared/glossary/#tiling): loading small blocks of each matrix into fast on-chip memory, multiplying them there, and reusing them before touching slow memory again. Writing this in [Triton](/shared/glossary/#triton) and aiming for >50% of [cuBLAS](/shared/glossary/#cublas) [throughput](/shared/glossary/#throughput) teaches why memory movement, not arithmetic, is the real cost.

## Why This Matters

Matrix multiplication dominates the runtime of almost every neural network, so understanding how a good matmul kernel is structured is the key to understanding GPU performance in general.
