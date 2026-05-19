# AI Hardware: From Beginner to Advanced

A comprehensive guide to the hardware that makes modern AI possible — from GPU and TPU architecture, through memory hierarchies, kernel-level programming, interconnects, quantization, and inference systems, to the realistic scope of "DIY AI hardware" projects. The aim is to take you from "I rent a GPU on RunPod and run `nvidia-smi`" to "I can read a tensor-core utilization profile, write a Triton kernel that hits >70% of peak bandwidth, design a multi-GPU rig, and tell when a hardware claim in a paper is real vs marketing."

> **An honest framing.** This guide covers AI hardware as it exists in 2026, not as it's marketed. Most of "DIY AI hardware" is *assembling and programming* hardware that already exists — building a multi-GPU workstation, writing custom CUDA/Triton kernels, fine-tuning quantization. *Fabricating your own GPU/TPU/HBM is not a thing for individuals* — single GPUs cost billions of dollars to design and tens of millions per wafer to manufacture. The achievable DIY frontier is custom inference ASICs via tinytapeout-style flows, FPGA acceleration, and serious system integration. This guide is honest about that distinction.

---

## Table of Contents

1. [Phase 0: Prerequisites and Setup](#phase-0-prerequisites-and-setup)
2. [Phase 1: How a Modern Computer Computes](#phase-1-how-a-modern-computer-computes)
3. [Phase 2: GPU Architecture, Inside Out](#phase-2-gpu-architecture-inside-out)
4. [Phase 3: The Memory Hierarchy — Where Your Time Actually Goes](#phase-3-the-memory-hierarchy--where-your-time-actually-goes)
5. [Phase 4: CUDA, Triton, and Writing Real Kernels](#phase-4-cuda-triton-and-writing-real-kernels)
6. [Phase 5: TPUs, NPUs, and Alternative Accelerators](#phase-5-tpus-npus-and-alternative-accelerators)
7. [Phase 6: Interconnects, Multi-GPU, and Multi-Node](#phase-6-interconnects-multi-gpu-and-multi-node)
8. [Phase 7: Numeric Formats and Quantization](#phase-7-numeric-formats-and-quantization)
9. [Phase 8: Inference Systems and Serving](#phase-8-inference-systems-and-serving)
10. [Phase 9: DIY AI Hardware — What's Actually Possible](#phase-9-diy-ai-hardware--whats-actually-possible)
11. [Phase 10: Frontier Topics](#phase-10-frontier-topics)
12. [Suggested Timeline](#suggested-timeline)
13. [Key Advice](#key-advice)
14. [Additional Resources](#additional-resources)
15. [Glossary](#glossary)

---

## Phase 0: Prerequisites and Setup

The prerequisites here are different from the other guides — you can be a less experienced ML practitioner and still get a lot out of this material, as long as you have systems-programming chops.

### Concepts You Should Already Know

- **C and basic C++** — enough to read CUDA code; you don't need template metaprogramming
- **Operating systems basics**: virtual memory, paging, processes vs threads, system calls
- **Computer architecture basics**: caches (L1/L2/L3), pipelining, SIMD, instruction-level parallelism — if these are blanks, do [CS:APP](https://csapp.cs.cmu.edu/) Ch. 5–6 first
- **PyTorch fluency**: you've trained models and profiled at least one — see the [PyTorch Deep Dive Guide](pytorch-deep-dive-guide.md)
- **Linear algebra at the FLOP level**: you can count matmul FLOPs in your head (2·M·N·K for an M×K times K×N matrix)
- **A modern GPU** — owned, rented, or borrowed. Cloud is fine. You will write code that runs.

### What You Need Installed (Recommended)

- **CUDA Toolkit 12.x+** if working with NVIDIA
- **Triton** (installed with PyTorch 2.x by default)
- **Nsight Systems (`nsys`)** and **Nsight Compute (`ncu`)** — NVIDIA's profilers; non-optional
- **`py-spy`** — for Python-side profiling
- **A profiler-friendly setup** — at least one machine where you can actually capture traces (cloud profiling is annoying)
- **Optional**: a Mac with Apple Silicon for MPS / MLX experiments; a small FPGA dev board (DE10-Nano, ULX3S) for Phase 9

### Hardware Reality, In Numbers

```
For context, as of 2026:

                         FP16/BF16 TFLOPs   HBM cap   HBM B/W    Power     ≈ Cost (cloud / used)
NVIDIA RTX 4090           ~330 (sparse)     24 GB     ~1.0 TB/s   450 W    $1.50–2.50/hr, ~$1.5k
NVIDIA RTX 5090           ~838 (sparse)     32 GB     ~1.8 TB/s   575 W    ~$2/hr, ~$2k
NVIDIA L40S               ~362 (sparse)     48 GB     ~864 GB/s   350 W    $1–1.5/hr
NVIDIA A100 (80 GB)       ~624              80 GB     ~2.0 TB/s   400 W    $1–2/hr, ~$10–15k used
NVIDIA H100 (SXM)         ~990              80 GB     ~3.35 TB/s  700 W    $2–4/hr
NVIDIA H200 (SXM)         ~990              141 GB    ~4.8 TB/s   700 W    $4–6/hr
NVIDIA B200 (SXM)         ~2,250            192 GB    ~8.0 TB/s   1000 W   $6–10/hr
NVIDIA GB200 (1 GPU half) similar to B200    192 GB    ~8.0 TB/s   1000 W   priced as systems
AMD MI300X                ~1,300            192 GB    ~5.3 TB/s   750 W    $2–4/hr
AMD MI325X                ~1,300+           256 GB    ~6.0 TB/s   1000 W   $3–5/hr
Google TPU v5p (1 chip)    ~459             95 GB     ~2.8 TB/s   ~250 W   Google Cloud only
Google TPU v6e (Trillium) ~918              32 GB     ~1.6 TB/s   ~250 W   Google Cloud only
Apple M4 Max (GPU)        ~17               unified   ~546 GB/s   ~50 W    consumer Macs

Specs change constantly; numbers above are spec-sheet peak.
Real workloads usually achieve 30–70% of peak; the rest of this guide
is largely about why and how to do better.
```

### Resources

- [CS:APP — Computer Systems: A Programmer's Perspective](https://csapp.cs.cmu.edu/) — if you skipped systems classes
- [GPU MODE lectures and Discord](https://github.com/gpu-mode/lectures) — the most active modern community for this material
- [Horace He — Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) — required reading, twice
- [NVIDIA developer documentation](https://docs.nvidia.com/cuda/) — your primary reference

---

## Phase 1: How a Modern Computer Computes

You can't understand GPUs without first understanding why CPUs aren't enough. This phase is the conceptual setup.

### Concepts to Learn

- **The memory wall**: the historical gap between CPU compute speed and memory bandwidth, and why it shaped everything that came after
- **CPU vs GPU philosophy**:
  - **CPU**: few powerful cores, deep pipelines, huge caches, branch prediction, out-of-order execution — optimized for **latency** on irregular code
  - **GPU**: thousands of simpler cores, shallow pipelines, tiny per-thread cache, massive parallelism — optimized for **throughput** on regular code
- **SIMD vs SIMT**:
  - **SIMD** (Single Instruction Multiple Data): one instruction, many data elements per core (AVX-512, NEON)
  - **SIMT** (Single Instruction Multiple Threads): one instruction, many threads — NVIDIA's variant; thread divergence and warps
- **Roofline model** — the most important diagram in the field. Performance is bounded by the *minimum* of (peak FLOPs) and (peak memory bandwidth × arithmetic intensity)
- **Arithmetic intensity**: FLOPs per byte of memory accessed. Higher = compute-bound. Lower = memory-bound. Most deep-learning ops are memory-bound
- **Amdahl's law** — your serial fraction limits you no matter how much parallelism you add

### The Roofline Diagram

```
   Performance (FLOPs/sec)
         │
   peak ─┼──────────────────────────────── ← compute roof
         │                       /  
         │                     /  
         │                   /  
         │                 / ← memory roof
         │               /     (slope = bandwidth, B/s)
         │             /
         │           /
         │         /
         │       /
         │     /
         │   /
         │ /
         └────────────────────────────── ── Arithmetic intensity (FLOPs/byte)
              ↑                  ↑
         Memory-bound        Compute-bound region
         (most DL kernels    (large matmul lives here)
          live here)

Kernels in the memory-bound region only get faster if you reduce memory traffic.
Adding FLOPs there is wasted.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Hand-counted FLOPs | Count the FLOPs in one transformer block (attention + MLP) for a given shape | ⭐⭐ |
| Roofline by hand | For 5 operations (matmul, layernorm, softmax, gelu, transpose), compute arithmetic intensity and predict which are memory-bound on an A100 | ⭐⭐⭐ |
| Bandwidth measurement | Measure the actual achieved memory bandwidth of a simple copy kernel on your GPU; compare to the spec | ⭐⭐⭐ |
| AVX-512 study | Write a vector sum in scalar, autovectorized, and intrinsic AVX-512 code; benchmark on a CPU | ⭐⭐⭐ |
| GPU vs CPU bake-off | Run the same matmul on CPU (NumPy) and GPU (PyTorch); report wall-clock and FLOPs/sec | ⭐⭐ |

### Sample Code: Arithmetic Intensity in Practice

```python
# A function returns A · B for square matrices of size N×N.
# FLOPs:   2·N³   (N³ multiplies + N³ adds)
# Bytes:   3·N²·dtype_bytes  (read A, read B, write C)
#
# AI = FLOPs / Bytes = (2·N³) / (3·N²·b) = (2/3)·N/b
#
# So for N=4096, fp16 (b=2):  AI = (2/3)·4096/2 ≈ 1365 FLOPs/byte
# An H100 needs AI ≥ ~300 FLOPs/byte to be compute-bound on bf16.
# → Big matmul is compute-bound (good news, that's the easy case).

# For a layernorm on the same data:
# FLOPs:   ~5·N   per row of length N (mean, variance, normalize)
# Bytes:   ~3·N·b (read input, write output, possibly read params)
# AI:      ~5N / (3Nb) = 5/(3·b) ≈ 0.83 FLOPs/byte for fp16
# → Layernorm is severely memory-bound. Fuse it.
```

### Key Insight

The single most important number in modern AI hardware is **the arithmetic intensity break-even point** for your hardware — typically 300–500 FLOPs/byte on a high-end GPU at bf16. Operations above that are compute-bound and benefit from more FLOPs; operations below are memory-bound and only get faster if you reduce memory traffic (fusion, lower precision, recomputation). Almost every "fast" deep learning paper (FlashAttention, mixed precision, kernel fusion, gradient checkpointing) is, at its core, a roofline argument. Internalize this and the rest of the field becomes legible.

### Resources

- [Roofline: An Insightful Visual Performance Model (Williams et al., 2009)](https://dl.acm.org/doi/10.1145/1498765.1498785) — the original paper, still relevant
- [Hennessy & Patterson — Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1) — the textbook
- [Horace He — Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)
- [Stephen Jones (NVIDIA) — "How GPU Computing Works"](https://www.youtube.com/watch?v=3l10o0DYJXg) — 50 minutes, exceptional

---

## Phase 2: GPU Architecture, Inside Out

GPUs are the dominant AI hardware. This phase is the irreducible mental model.

### Concepts to Learn

- **The GPU stack**: GPU → Graphics Processing Cluster (GPC) → Streaming Multiprocessor (SM) → warps → threads
- **Streaming Multiprocessor (SM)** — the GPU's "core"; contains scheduling, registers, shared memory, and execution units. Modern Hopper has 132 SMs; Blackwell B200 has 148
- **Warp** — 32 threads scheduled together; the unit of execution. Threads in a warp execute the same instruction in lockstep (or stall together on divergence)
- **Tensor Cores** — specialized matrix-multiply units; do a 4×4 (or 8×8, or 16×16) matmul per instruction. The reason an H100 does ~1 PFLOP at FP8 vs ~67 TFLOPs at FP32
- **Compute capability** — NVIDIA's version number for GPU features (8.0 = Ampere/A100, 9.0 = Hopper/H100, 10.0 = Blackwell). Determines what Tensor Core sizes and dtypes are available
- **Warp scheduling** — multiple warps per SM, the SM swaps between them to hide memory latency. Higher *occupancy* (more active warps) usually means more latency hiding
- **Thread divergence** — when threads in a warp take different branches, both branches run serially. Major performance loss
- **The execution hierarchy in your kernel code**:
  - **Thread** — a single execution context
  - **Block / CTA** — a group of threads (up to 1024) that share memory and can synchronize
  - **Grid** — all blocks launched by a kernel
- **Asynchronous engines**: DMA, copy engines — many things happen in parallel with compute

### The NVIDIA GPU Hierarchy, Visualized

```
GPU (one chip)
├── HBM (high-bandwidth memory) — 80–192 GB
├── L2 cache (40–60 MB, shared across all SMs)
└── 100+ Streaming Multiprocessors (SMs)
    ├── Each SM contains:
    │   ├── Tensor Cores (the FLOPs)
    │   ├── CUDA cores (for non-matmul ops)
    │   ├── Special function units (sin, exp, rsqrt)
    │   ├── L1 cache + shared memory (~ 100–230 KB, configurable split)
    │   ├── Register file (64K 32-bit registers)
    │   └── Warp schedulers (4 per SM on Hopper)
    │
    └── Each SM runs many warps simultaneously
        └── Each warp = 32 threads in lockstep

You launch a grid of blocks. Each block is assigned to an SM and stays there.
The SM time-slices between resident warps to hide memory latency.
```

### Generation-by-Generation Highlights (NVIDIA)

```
Volta (V100, 2017)     First Tensor Cores (FP16 matmul)
Turing (RTX 20xx)      Tensor Cores in consumer cards; RT cores for ray tracing
Ampere (A100, 2020)    BF16 support; structured sparsity; MIG (multi-instance)
Hopper (H100, 2022)    FP8 (E4M3, E5M2); Transformer Engine; Thread Block Clusters
Hopper-Next (H200)     Same compute as H100; ~1.4× HBM bandwidth, ~1.8× HBM capacity
Blackwell (B100/B200)  FP4; 2× tensor core FLOPs vs Hopper; NVL72 systems
Blackwell-Ultra (B300) Refresh; more HBM, higher clocks
Rubin (announced)      Next-gen; production 2026–2027
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| `nvidia-smi` deep dive | For your GPU: report exact compute capability, SM count, HBM type/capacity/bandwidth, NVLink topology | ⭐ |
| Tensor core utilization | Run `nsys` and `ncu` on a matmul; observe % of time tensor cores are active | ⭐⭐⭐ |
| Occupancy study | Vary block size in a simple CUDA kernel; observe SM occupancy in Nsight | ⭐⭐⭐ |
| Divergence demo | Write a kernel with intentional thread divergence; measure the slowdown | ⭐⭐⭐ |
| Spec compare | Pick three GPUs (consumer/datacenter/old); compute peak bf16 FLOPs, HBM B/W, and AI break-even point | ⭐⭐ |

### Key Insight

Most ML practitioners think of a GPU as a fast matmul engine. The reality is more interesting: a modern GPU is a *latency-hiding throughput machine*. The Tensor Cores deliver the FLOPs, but the warp scheduler is the unsung hero — it keeps the SM busy by switching between dozens of in-flight warps whenever one stalls on memory. This is why GPUs need high occupancy to perform well: an under-occupied SM is one that has nothing to switch to when a warp stalls, and it sits idle waiting for HBM. Understanding this changes how you read every profile.

### Resources

- [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [NVIDIA B200/Blackwell Whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture)
- [Inside Volta: The World's Most Advanced Data Center GPU (NVIDIA blog)](https://developer.nvidia.com/blog/inside-volta/) — older but the architecture intro is timeless
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — start at "Hardware Implementation"

---

## Phase 3: The Memory Hierarchy — Where Your Time Actually Goes

If Phase 2 is the FLOPs, Phase 3 is the bytes. In practice, this phase matters more.

### Concepts to Learn

- **The hierarchy** (high to low bandwidth, small to large capacity):
  1. **Registers** — per-thread, ~10 cycles, ~tens of KB per SM
  2. **Shared memory / L1** — per-SM, ~30 cycles, ~100–230 KB per SM
  3. **L2 cache** — chip-wide, ~200 cycles, ~40–60 MB
  4. **HBM** — chip-wide, ~400+ cycles, 80–192 GB
  5. **NVLink to other GPUs** — ~600 GB/s on Hopper, ~1800 GB/s on Blackwell
  6. **PCIe to CPU** — ~32 GB/s (Gen4) to ~64 GB/s (Gen5)
  7. **Networked storage** — varies; usually disastrously slow
- **HBM (High-Bandwidth Memory)**:
  - Stacked DRAM dies connected by through-silicon vias (TSVs)
  - HBM3: ~3.35 TB/s per chip (H100). HBM3e: ~4.8 TB/s (H200). HBM4 coming
  - Why it's expensive: complex packaging, limited suppliers (SK Hynix, Samsung, Micron)
  - Capacity is the new constraint: HBM is the fixed-resource bottleneck for big-model training
- **Memory coalescing** — adjacent threads in a warp accessing adjacent memory addresses can be combined into one transaction. Non-coalesced access cratrers bandwidth
- **Bank conflicts** — shared memory is split into 32 banks; two threads accessing the same bank serialize
- **L2 hit rate** — surprisingly important on modern GPUs; influences whether a memory-bound kernel reaches its roofline
- **Tile sizes and re-use** — the whole point of tiling is to load data once and reuse it many times within a tile; this is the source of arithmetic intensity for matmul
- **Memory bandwidth in practice** — almost no kernel hits theoretical peak; 70–80% is excellent

### Memory Latencies, Visualized

```
Registers              ~10 cycles      ~ free
Shared memory / L1     ~30 cycles      ~3× register
L2 cache               ~200 cycles     ~20× register
HBM                    ~400+ cycles    ~40× register
NVLink (peer GPU)      ~1500 cycles    ~150× register
PCIe (CPU)             ~30,000 cycles  ~3000× register

Each step down is roughly an order of magnitude.
The whole game is to keep your working set as high in this hierarchy
as you can for as long as you can.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Coalesced vs non-coalesced | Write two CUDA kernels — adjacent-stride vs strided access — and measure bandwidth | ⭐⭐⭐ |
| Bank conflict demo | Reproduce a shared-memory bank conflict; report the slowdown | ⭐⭐⭐⭐ |
| Tile size sweep | Implement a tiled matmul; vary tile size; observe the sweet spot | ⭐⭐⭐⭐ |
| HBM saturation | Write a vector-add and measure HBM bandwidth utilization; tune until > 80% of peak | ⭐⭐⭐⭐ |
| L2 hit rate analysis | Use Nsight Compute to inspect L2 hit rate of an attention kernel; explain what you see | ⭐⭐⭐⭐ |

### Sample Code: A Coalesced vs Strided Load Demo (CUDA)

```cuda
// Coalesced: adjacent threads load adjacent floats. One memory transaction.
__global__ void coalesced(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

// Strided: each thread loads from a stride-32 location. Up to 32 transactions
// per warp. Bandwidth collapses ~10–30×.
__global__ void strided(const float* in, float* out, int n, int stride) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (i < n) out[i] = in[i] * 2.0f;
}
```

### Key Insight

The phrase "GPUs are fast at matrix multiplication" is misleading. GPUs are fast at *operations that have enough arithmetic intensity to keep the Tensor Cores busy without exhausting memory bandwidth*. Matrix multiplication happens to have this property at large enough sizes. Many other operations don't — and those are the ones you have to write or fuse carefully. Once you internalize that the memory hierarchy is what's actually being optimized, you stop being surprised that "small" optimizations (tiling, fusion, recomputation) produce 5–10× speedups; they're not micro-optimizations, they're moving the working set up the hierarchy.

### Resources

- [NVIDIA — Best Practices Guide: Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [FlashAttention paper (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — the textbook example of memory-hierarchy thinking
- [Lei Mao — CUDA Coalesced Memory Access](https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/)
- [Stephen Jones — Memory: GPUs and CPUs (talk)](https://www.youtube.com/watch?v=4APDDuL34Tk)

---

## Phase 4: CUDA, Triton, and Writing Real Kernels

The point at which you stop being a hardware consumer and become a producer.

### Concepts to Learn

- **CUDA C++ basics**: `__global__`, `__device__`, thread/block IDs, `__syncthreads()`, shared memory
- **Launch configuration**: choosing block and grid dimensions; occupancy calculator
- **CUDA streams**: how multiple operations run concurrently; events for synchronization
- **CUB and Thrust**: the standard library of CUDA — already-optimized parallel primitives
- **cuBLAS and cuDNN**: NVIDIA's optimized libraries; "compete with these on matmul/attention only if you have a very good reason"
- **Triton** — Python-flavored GPU language; pointers, blocks, masks, `tl.load`/`tl.store`, autotuning
- **The dispatcher integration**: how a custom kernel becomes a `torch.library.custom_op` that `torch.compile` can pick up
- **CUTLASS** — NVIDIA's template library for matmul-like kernels; what FlashAttention is built on
- **The kernel-writing workflow**: prototype in Python (PyTorch eager), profile, identify a memory-bound region, rewrite the offending op as one fused kernel, benchmark, regress

### When to Write a Custom Kernel

```
You should NOT write a custom kernel if:
  - cuBLAS / cuDNN / Flash-Attn already does this
  - torch.compile already fuses this acceptably
  - You haven't profiled to confirm this op is the bottleneck

You SHOULD write a custom kernel if:
  - You've profiled and a specific op is memory-bound AND non-fusable
  - You can fuse several ops together to save memory traffic
  - You need a numerically-different version (stable softmax variant, etc.)
  - You're implementing a research idea not expressible in standard ops
  - You need a precision/format that isn't supported (FP6, FP4, custom)
```

### Triton vs CUDA: When to Pick Which

```
Pick Triton when:
  - You're prototyping a kernel and want fast iteration
  - The kernel is "natural" in block-level operations (matmul, attention)
  - You want autotuning across multiple shapes
  - You don't need every last % of peak

Pick CUDA when:
  - You need very tight control of warp-level operations
  - You're integrating with cuBLAS / CUTLASS / cuDNN
  - You're targeting features Triton doesn't expose yet
  - You need to squeeze out the last 5–10%
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| CUDA vector add | Write the canonical "hello world" CUDA kernel; profile and tune block size | ⭐⭐ |
| CUDA tiled matmul | Implement a shared-memory tiled matmul; aim for > 50% of cuBLAS throughput | ⭐⭐⭐⭐ |
| Triton softmax | Implement softmax in Triton; verify numerical match and bandwidth utilization | ⭐⭐⭐ |
| Triton matmul | Tile-based matmul in Triton; aim for > 70% of cuBLAS | ⭐⭐⭐⭐ |
| Fused LayerNorm | One Triton kernel that does layernorm + linear; benchmark against eager + cuDNN | ⭐⭐⭐⭐ |
| Mini FlashAttention | Tiled, online-softmax attention kernel in Triton; verify numerical match | ⭐⭐⭐⭐⭐ |
| Custom op registration | Wrap your Triton kernel as a `torch.library.custom_op` so `torch.compile` can use it | ⭐⭐⭐⭐ |

### Sample Code: A Triton Fused LayerNorm Forward

```python
import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_fwd_kernel(
    X, Y, W, B,                 # input, output, weight, bias
    M, N,                       # rows, cols
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptrs = X + row * N + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    y = centered * rstd * w + b

    tl.store(Y + row * N + cols, y, mask=mask)

def layernorm(x, w, b, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    layernorm_fwd_kernel[(M,)](x, y, w, b, M, N, eps, BLOCK_N=BLOCK_N)
    return y
```

### Key Insight

The two most expensive lines a GPU programmer writes are not `__global__` and `tl.dot`. They are the ones that move data: `tl.load` and `tl.store`. Every kernel you write should be designed around minimizing these — load data once, compute as much with it as you can while it's in registers or shared memory, then write the result. FlashAttention is famous because it took the same FLOPs as standard attention and reorganized them so that each input element is read from HBM exactly once. That's the entire trick. Once you see this pattern, you start spotting it everywhere — fused MLPs, paged attention, every "X but faster" paper since 2022.

### Resources

- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) — start at vector add, end at FlashAttention
- [CUTLASS](https://github.com/NVIDIA/cutlass) — the matmul template library
- [GPU MODE lectures](https://github.com/gpu-mode/lectures) — best community for kernel work
- [PyTorch Deep Dive Guide — Phase 6](pytorch-deep-dive-guide.md#phase-6-custom-kernels--c-cuda-and-triton-extensions)
- [Lei Mao's blog](https://leimao.github.io/) — clear walkthroughs of CUDA topics

---

## Phase 5: TPUs, NPUs, and Alternative Accelerators

NVIDIA dominates, but the rest of the landscape matters — both because non-NVIDIA hardware is real and because the architectural variations teach you something.

### Concepts to Learn

- **Google TPUs**:
  - **Systolic array** — the core idea: a 2D grid of multiply-add units where data flows through; matmul-native, no caches in the traditional sense
  - **MXU (Matrix Multiply Unit)** — typically 128×128 or 256×256 on modern TPUs
  - **HBM-attached but lower bandwidth per chip than NVIDIA** — compensated by huge pods (TPU v5p: up to 8960 chips in one pod)
  - **The pod is the unit** — TPUs are designed to be programmed as one giant accelerator with thousands of chips
  - **XLA (compiler)** — TPUs are programmed via XLA, not anything like CUDA. JAX and PyTorch/XLA are the entry points
  - **Generations**: TPU v3, v4, v5e, v5p, v6e (Trillium), v7 (Ironwood, announced 2024–2025)
- **AMD CDNA / RDNA**:
  - **MI300X / MI325X** — datacenter accelerators competing with H100/H200
  - **ROCm** — the software stack; getting better but still not on par with CUDA
  - **HIP** — AMD's CUDA-compatible programming layer; "port your CUDA code"
- **Apple Silicon**:
  - **Unified memory** — GPU and CPU share the same memory pool, no copies; great for inference
  - **MPS backend** for PyTorch; **MLX** as the native ML framework
  - The Mac Studio with M-series Max/Ultra has become a real local-inference platform
- **Intel Gaudi / Habana**:
  - Gaudi 2, 3 — competitive on TCO for inference; ecosystem weak
- **Cerebras**:
  - **Wafer-scale** — one chip = one entire wafer = ~850k cores, 40GB SRAM on chip
  - Niche but interesting; targets specific workloads where its huge on-chip memory shines
- **Groq**:
  - **LPU (Language Processing Unit)** — designed specifically for low-latency LLM inference
  - Extremely fast for batch-size-1 LLM generation; less competitive at high throughput
- **Sambanova, Tenstorrent, Esperanto, Etched**:
  - Tenstorrent (Wormhole, Blackhole, Grayskull) — open developer cards; the most accessible non-NVIDIA option for individuals
  - Etched (Sohu) — transformer-specific ASIC announced 2024
- **Edge accelerators**:
  - **NVIDIA Jetson** family — robotics/edge inference
  - **Apple Neural Engine** — phones, iPads, Macs
  - **Qualcomm Hexagon NPU**, **Google Edge TPU (Coral)**, **MediaTek APU** — phone-class NPUs

### Systolic Array vs SIMT, Visualized

```
NVIDIA-style SIMT (per SM):
   - thousands of threads, each doing matmul fragments via Tensor Cores
   - explicit data movement through shared memory + L1 + registers
   - flexible: also does softmax, layernorm, attention, etc.

TPU-style systolic array:
   ┌─────────────────────────────────────┐
   │  matrix B flows down                │
   │  ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓                    │
   │  □ □ □ □ □ □ □ □  ← matrix A flows  │
   │  □ □ □ □ □ □ □ □     right →        │
   │  □ □ □ □ □ □ □ □                    │
   │  □ □ □ □ □ □ □ □  Each □ is a MAC.  │
   │  □ □ □ □ □ □ □ □  Accumulators stay │
   │  □ □ □ □ □ □ □ □  in place; data    │
   │  □ □ □ □ □ □ □ □  flows through.    │
   │  □ □ □ □ □ □ □ □                    │
   └─────────────────────────────────────┘
   - data moves through the array; no caching needed
   - extreme matmul efficiency
   - less flexible: softmax/layernorm done by separate vector units
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Run a TPU notebook | Free Kaggle/Colab TPU; train a small model with JAX or PyTorch/XLA | ⭐⭐⭐ |
| AMD MI300 inference | If you can rent one: run a large LLM on MI300X via ROCm + vLLM; report tokens/sec | ⭐⭐⭐⭐ |
| Apple Silicon LLM | Run a 7B–70B model on a Mac Studio via MLX or llama.cpp; benchmark | ⭐⭐⭐ |
| Compare accelerators | Same model, same batch, on three different hardware types (e.g., A100, M2 Max, free TPU); report all metrics | ⭐⭐⭐⭐ |
| Tenstorrent dev | Buy or rent a Tenstorrent card; write a simple kernel using their open SDK | ⭐⭐⭐⭐⭐ |

### Key Insight

The interesting design space is not just "more FLOPs." The TPU's bet is that *matmul is enough*, and everything else (memory hierarchy, attention, layernorm) can be designed around that simplification. Cerebras's bet is that *moving data off-chip is the bottleneck*, so use a wafer's worth of SRAM. Groq's bet is that *latency, not throughput, is the production constraint*, so build a deeply pipelined chip optimized for tokens-per-user-per-second. None of these has yet beaten NVIDIA in the broad market, but each illustrates a real architectural axis. Reading their architecture papers is genuine education even if you only ever ship on NVIDIA.

### Resources

- [Google TPU paper (Jouppi et al., 2017)](https://arxiv.org/abs/1704.04760) — the original
- [TPU v4 paper (Jouppi et al., 2023)](https://arxiv.org/abs/2304.01433)
- [AMD MI300 architecture](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
- [Apple MLX](https://github.com/ml-explore/mlx)
- [Tenstorrent open SDK](https://github.com/tenstorrent/tt-metal)
- [Groq architecture paper](https://groq.com/wp-content/uploads/2020/06/ISCA-TSP.pdf)

---

## Phase 6: Interconnects, Multi-GPU, and Multi-Node

A single GPU is increasingly insufficient. This phase is about how you scale.

### Concepts to Learn

- **PCIe** — the standard CPU-GPU and (in some setups) GPU-GPU link. Gen4 ~32 GB/s per direction; Gen5 ~64 GB/s; Gen6 ~128 GB/s (rolling out 2025–2026)
- **NVLink** — NVIDIA's fast GPU-GPU interconnect. H100: 900 GB/s total; B200: ~1.8 TB/s total. Within a node, much faster than PCIe
- **NVSwitch** — NVLink switch chip; lets a node of 8 GPUs talk all-to-all at full NVLink speed
- **NVL72** — Blackwell rack-scale system: 72 GPUs in one NVLink domain. Effectively one giant GPU for software purposes
- **InfiniBand (IB)** — the standard high-speed network for HPC and AI clusters; 400 Gb/s NDR, 800 Gb/s XDR coming. Sub-microsecond latency
- **RoCE (RDMA over Converged Ethernet)** — bringing RDMA semantics to Ethernet; what hyperscalers usually deploy
- **NCCL (NVIDIA Collective Communications Library)** — the library that does AllReduce/AllGather/ReduceScatter over NVLink/IB. Topology-aware
- **Collective operations** — AllReduce, AllGather, ReduceScatter, Broadcast, AllToAll; the building blocks of distributed training
- **Tree vs ring collectives**: ring is bandwidth-optimal but latency-bad at small messages; tree is the opposite. NCCL picks based on message size and topology
- **Network topology**: fat-tree, dragonfly, rail-optimized; matters for very large clusters
- **Optical interconnects** — coming for inter-rack; co-packaged optics for in-rack at the long horizon

### Bandwidth Hierarchy in a 2026 H100/B200 System

```
Within an SM (registers, shared mem)   → 10–20 TB/s (each)
Within a GPU (L2 cache, HBM)           → 3–8 TB/s
Between GPUs in same node (NVLink)     → 900 GB/s – 1.8 TB/s
Between nodes (InfiniBand NDR)         → 400 Gb/s = 50 GB/s
                                       → 800 Gb/s = 100 GB/s with XDR
Between datacenters                    → highly variable, usually disastrous

A factor of ~10 at every step except the first.
If your distributed training plan has GPUs talking to each other,
think hard about which step in this hierarchy you're going through.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| `nccl-tests` | Run NVIDIA's official NCCL benchmark on your multi-GPU setup; report AllReduce bandwidth | ⭐⭐⭐ |
| Multi-GPU DDP | Train a model on 2+ GPUs with DDP; profile to see AllReduce time vs compute time | ⭐⭐⭐ |
| FSDP scaling | Scale to 4–8 GPUs with FSDP; observe how communication scales with world size | ⭐⭐⭐⭐ |
| Multi-node setup | Set up 2-node training (cloud or two boxes on a LAN); report cross-node bandwidth | ⭐⭐⭐⭐ |
| Topology study | Use `nvidia-smi topo -m` to inspect your GPU topology; predict NCCL performance based on it | ⭐⭐⭐ |

### Key Insight

In a 2026 frontier training cluster, the network is *the* design constraint. Every GPU-GPU link, every InfiniBand cable, every switch hop has a real cost in microseconds and real-dollar terms. For 100k-GPU clusters, the network can be more expensive than the GPUs. The hyperscaler arms race for "AI compute" is largely a network engineering competition disguised as a GPU competition. From an individual practitioner's perspective, what this means is: when training spans nodes, profile the communication explicitly, choose your parallelism strategy (DP/TP/PP/FSDP) to minimize cross-node communication, and don't fight your network topology.

### Resources

- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [`nccl-tests`](https://github.com/NVIDIA/nccl-tests)
- [NVIDIA NVLink and NVSwitch overview](https://www.nvidia.com/en-us/data-center/nvlink/)
- [Mellanox / NVIDIA Networking docs](https://docs.nvidia.com/networking/)
- [PyTorch Deep Dive Guide — Phase 7](pytorch-deep-dive-guide.md#phase-7-distributed-training--ddp-fsdp-and-beyond)

---

## Phase 7: Numeric Formats and Quantization

The hardware game is largely the precision game. This phase is about what bits to use and when.

### Concepts to Learn

- **Floating-point primer**: sign + exponent + mantissa; what each does
- **Common formats**:
  - **FP32** — 1+8+23 — the dinosaur; almost never used for training in 2026
  - **FP16** — 1+5+10 — narrow range, needs `GradScaler`; mostly being replaced
  - **BF16** — 1+8+7 — same range as FP32, less precision; the modern training default
  - **TF32** — NVIDIA-only marketing name for "FP32 input, 10-bit mantissa internally"; basically transparent
  - **FP8** — multiple variants (E4M3, E5M2); the modern *inference* default and increasingly training
  - **FP4** — Blackwell-introduced; 1+2+1 or 1+3+0 variants; pushed mostly for inference
  - **INT8 / INT4** — integer quantization, no fractional part; common for inference
  - **NF4** — "normal-float 4-bit" from QLoRA; non-uniform quantization grid tuned for neural-net weights
- **Range vs precision trade-off**: more exponent bits = bigger range = harder to overflow; more mantissa bits = better precision near each value
- **Quantization fundamentals**:
  - **PTQ (Post-Training Quantization)** — quantize an already-trained model; cheap, modest quality loss
  - **QAT (Quantization-Aware Training)** — train with simulated quantization; better quality, expensive
  - **Calibration data** — sample inputs used to set quantization scales
  - **Per-tensor** vs **per-channel** vs **per-group** quantization scales
  - **Activation quantization** vs **weight-only quantization** — weight-only is much easier; activations need careful range handling
- **Modern quantization methods**:
  - **GPTQ** — gradient-free, weight-only, INT4; the workhorse
  - **AWQ** — preserves "important" weights at higher precision
  - **SmoothQuant** — handles activation outliers
  - **GGUF / GGML quantizations** — llama.cpp's family; K-quants, I-quants
- **FP8 training** — Hopper's Transformer Engine handles dynamic scaling automatically; works well for many models
- **The KV-cache** quantization angle — quantizing the KV cache can dramatically reduce memory for long-context inference

### The Format Zoo, Visualized

```
                Sign  Exp  Mantissa   Range          Precision
                ──── ──── ────────   ─────────────  ─────────────
FP32             1    8    23        ~10⁻³⁸ … 10³⁸  ~7 dec digits
TF32             1    8    10        same as FP32   ~3 dec digits
BF16             1    8    7         ~10⁻³⁸ … 10³⁸  ~2 dec digits
FP16             1    5    10        ~10⁻⁵ … 10⁵    ~3 dec digits ← narrow!
FP8 E4M3         1    4    3         ~10⁻³ … 10²    ~1 dec digit
FP8 E5M2         1    5    2         ~10⁻⁵ … 10⁴    ~1 dec digit (wider range)
FP4 E2M1         1    2    1         tiny           extremely lossy
INT8             1    -    7         -128…127       integer
INT4             1    -    3         -8…7           integer (or unsigned)
NF4              -    -    -         non-uniform    4-bit "buckets"
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Format sweep | Train the same model in FP32, BF16, FP16, FP8; compare quality and speed | ⭐⭐⭐ |
| Quantize a small LLM | Take a 7B model; quantize to INT4 with GPTQ; measure quality drop (perplexity, MMLU subset) | ⭐⭐⭐ |
| KV-cache quantization | Quantize a model's KV cache to INT8; measure long-context inference savings | ⭐⭐⭐⭐ |
| Calibration data study | Quantize with 1, 16, 256 calibration samples; measure how quality varies | ⭐⭐⭐⭐ |
| Per-channel vs per-tensor | For one model: try both quantization granularities; measure quality and inference speed | ⭐⭐⭐⭐ |
| QLoRA fine-tune | Fine-tune a 7B model with QLoRA on a single 24 GB GPU; measure peak memory | ⭐⭐⭐⭐ |

### Sample Code: A Simple Symmetric Quantizer

```python
import torch

def quantize_symmetric_int8(x: torch.Tensor, dim=None):
    """Symmetric per-tensor or per-channel INT8 quantization."""
    if dim is None:
        amax = x.abs().max()
    else:
        amax = x.abs().amax(dim=dim, keepdim=True)
    scale = amax / 127.0
    q = torch.clamp((x / scale).round(), -128, 127).to(torch.int8)
    return q, scale

def dequantize(q: torch.Tensor, scale: torch.Tensor):
    return q.to(torch.float32) * scale

# Quantization error measurement
x = torch.randn(1024, 1024)
q, s = quantize_symmetric_int8(x, dim=1)        # per-row
x_dq = dequantize(q, s)
err = (x - x_dq).abs().mean()
print(f"mean abs error: {err.item():.5f}")     # ~0.002 for normally-distributed input
```

### Key Insight

The hardware FLOP curve has been climbing dramatically partly because of better silicon and partly because **the precision keeps shrinking**. A 2026 Blackwell at FP4 has ~30× the headline FLOPs of a 2017 V100 at FP16. Maybe 5× of that is real compute; the rest is bit-shrinkage. The catch: every bit you give up has to be earned back somehow — by calibration, by careful scaling, by smarter algorithms. The "FP4 LLM" you're running on Blackwell isn't free; it's the cumulative product of dozens of papers and years of hardware-software co-design. The lesson for practitioners: precision is a knob, not a property. Tune it intentionally.

### Resources

- [GPTQ paper](https://arxiv.org/abs/2210.17323)
- [AWQ paper](https://arxiv.org/abs/2306.00978)
- [SmoothQuant paper](https://arxiv.org/abs/2211.10438)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) — FP8 training
- [llama.cpp quantization docs](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)

---

## Phase 8: Inference Systems and Serving

Training is half the field; serving is the other half. The hardware constraints differ.

### Concepts to Learn

- **Inference economics** — at scale, the cost of inference dominates the cost of training. Optimizing serving is the highest-leverage hardware work in commercial AI
- **Latency vs throughput** — different goals require different optimizations
  - **TTFT (Time To First Token)** — user-facing latency; dominated by prefill
  - **TPOT (Time Per Output Token)** — steady-state generation speed
  - **TBT (Time Between Tokens)** — inter-token jitter
- **Prefill vs decode** — the two phases of LLM inference:
  - **Prefill** — process the prompt all at once; compute-bound; high arithmetic intensity
  - **Decode** — generate tokens one at a time; memory-bound; AI ~ 1 FLOP per byte of weights
  - The asymmetry is so stark that some systems use *different hardware* for the two phases
- **KV cache** — the cached key/values from previous tokens; grows linearly with context length; often dominates inference memory at long context
- **PagedAttention (vLLM)** — manage the KV cache like virtual memory pages; eliminates fragmentation, lets you batch wildly different sequence lengths
- **Continuous batching** — dynamically add/remove sequences from the batch as they complete or start; doubles throughput easily
- **Speculative decoding** — use a small "draft" model to propose tokens; the big model verifies. Up to 2–3× speedup with no quality loss
- **Disaggregated serving** — separate prefill and decode onto different GPUs/nodes; pioneered by 2024+ systems (DistServe, Splitwise)
- **Modern serving stacks**:
  - **vLLM** — open, dominant, fast-moving
  - **SGLang** — strong on complex structured generation
  - **TensorRT-LLM** — NVIDIA-optimized, lower-level
  - **TGI (Text Generation Inference)** — Hugging Face's stack
  - **llama.cpp** — CPU/edge; cross-platform, GGUF formats
- **Distillation** — train a smaller model to mimic a larger one; the most reliable way to speed up inference

### The Prefill/Decode Asymmetry

```
PREFILL  (process prompt of length L):
   FLOPs:           ~2 · L · model_params
   Memory traffic:  ~ model_params (weights read once)
   AI:              ~2L FLOPs / byte → COMPUTE-BOUND for L > ~512
   → Tensor cores happy. Hardware utilized.

DECODE  (generate one token):
   FLOPs:           ~2 · model_params
   Memory traffic:  ~ model_params (weights read once per token!) + KV cache
   AI:              ~2 FLOPs / byte → SEVERELY MEMORY-BOUND
   → Tensor cores idle. HBM bandwidth is the wall.

This is why a $40k H100 generating tokens hits maybe 5–10% of its peak FLOPs.
This is why batching helps so much (amortize the weight read across more queries).
This is why quantizing weights helps so much for decode but barely for prefill.
This is the single biggest economic fact about LLM serving.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Deploy with vLLM | Serve a 7B model with vLLM; measure tokens/sec at various batch sizes | ⭐⭐ |
| Latency vs throughput | Plot tokens/sec vs batch size; identify the knee | ⭐⭐⭐ |
| KV-cache memory math | For Llama-3 8B at context length 32k, batch 1, 8, 32: compute KV cache size; verify with `nvidia-smi` | ⭐⭐⭐ |
| Quantization for serving | Serve the same model in FP16, FP8, INT4 (AWQ/GPTQ); compare throughput and quality | ⭐⭐⭐⭐ |
| Speculative decoding | Set up a 7B model with a 0.5B draft model; measure speedup on a real benchmark | ⭐⭐⭐⭐ |
| Continuous batching demo | Compare static vs continuous batching on a workload with varying sequence lengths | ⭐⭐⭐ |

### Key Insight

LLM inference at scale is mostly an exercise in turning a fundamentally memory-bound operation into a less-memory-bound one. Every major serving innovation since 2022 — PagedAttention, continuous batching, speculative decoding, FP8/INT4 weights, KV-cache quantization, prefill/decode disaggregation — is, at its core, a different attack on the same problem: the weight matrix has to be read from HBM faster than HBM bandwidth allows. The next decade of inference engineering will be the same fight on increasingly clever ground.

### Resources

- [vLLM paper (PagedAttention)](https://arxiv.org/abs/2309.06180)
- [Splitwise paper (prefill/decode disaggregation)](https://arxiv.org/abs/2311.18677)
- [Speculative Decoding (Leviathan et al.)](https://arxiv.org/abs/2211.17192)
- [vLLM docs](https://docs.vllm.ai/)
- [SGLang](https://github.com/sgl-project/sglang)
- [TensorRT-LLM docs](https://docs.nvidia.com/tensorrt-llm/)

---

## Phase 9: DIY AI Hardware — What's Actually Possible

The honest section. Most "DIY AI hardware" content online is misleading; this phase is the calibration.

### What You Cannot Realistically DIY

- **Design and fabricate a new GPU/TPU/HBM from scratch.** A modern accelerator chip costs $200M–$2B to design, and each tapeout at TSMC is tens of millions of dollars. HBM stacks require specialized DRAM fabs (only SK Hynix, Samsung, Micron). No individual or small team is doing this.
- **Compete with NVIDIA on a general-purpose accelerator.** Many companies have tried; few have succeeded. The compounded software ecosystem is harder to replicate than the silicon.

### What You Can Realistically DIY

| Project class | What's involved | Realism |
|---------------|-----------------|---------|
| **Multi-GPU workstation build** | Pick GPUs, motherboard, PSU, cooling; assemble; benchmark | ⭐ Very realistic; ~$3–30k |
| **Power and thermals engineering** | Undervolting, cooling design, dual-PSU setups; running 2-4 RTX cards | ⭐⭐ Realistic with care |
| **Custom CUDA/Triton kernels** | Phase 4 territory; ship a kernel that beats cuBLAS on a niche shape | ⭐⭐⭐ Realistic; weeks of work |
| **FPGA inference accelerator** | Implement a small Transformer block on an FPGA dev board (Alveo, Pynq) | ⭐⭐⭐⭐ Realistic but slow ROI |
| **Tenstorrent kernel development** | Buy a Grayskull/Wormhole card; write programs against their open SDK | ⭐⭐⭐ Realistic, growing ecosystem |
| **Edge / Jetson deployment** | Quantize and deploy on Jetson Nano/Orin or Coral; real product territory | ⭐⭐ Very realistic |
| **Tinytapeout-style small ASIC** | Submit a tiny digital design that gets fabbed in a shared multi-project wafer | ⭐⭐⭐⭐⭐ Realistic but for tiny designs only |

### Concepts to Learn (for each path)

- **Multi-GPU workstation**: power budget math (your GPUs draw 350–600W each — your PSU and circuit need headroom), PCIe lane requirements (×16/×16 for 2-GPU; threadripper/Xeon for 4+ GPUs at full bandwidth), thermals (open-air vs blower-style cards; data-center-pulled cards often have blowers but need fan adapters), VRAM math (24 GB × 2 ≠ 48 GB unless NVLink-bridged on Ampere)
- **FPGAs**: bitstream concepts; Vivado/Vitis; Xilinx/AMD vs Intel/Altera; high-level synthesis (HLS) vs Verilog; the brutal reality that an FPGA at ~$2k might match a $400 GPU at ~10× the dev time
- **Tinytapeout / Skywater130**: open-source PDK (Skywater130nm); the multi-project wafer model; submitting a small design that gets fabricated in a shared run
- **Edge deployment**: TensorRT, ExecuTorch, ONNX Runtime; INT8 calibration on real edge devices; thermal throttling

### A Realistic 2-GPU AI Workstation Build (2026 mid-range)

```
GPU:          2× RTX 5090 (32 GB each, ~$2k–2.5k each)         $4,000–5,000
              ── or 2× used RTX A6000 (48 GB each) for VRAM    $7,000–9,000
              ── or 1× used H100 PCIe (80 GB)                  $20,000–25,000

CPU:          Threadripper 7960X (24 cores, 8-channel mem)     $1,400
              (PCIe lanes: 88 → both GPUs at ×16 + NVMe + 10GbE)

Motherboard:  Threadripper-class WS / sTR5 WRX90               $900
RAM:          128 GB DDR5 ECC, 8 DIMMs                         $800
Storage:      2 TB NVMe Gen5 + 4 TB Gen4 for datasets          $500
PSU:          1600W 80+ Platinum (dual-PSU possible for >1800W) $400
Case:         Open frame or full tower with good airflow       $300
Cooling:      Beefy CPU AIO + plenty of case fans              $300

Total (2× RTX 5090):                                           ~$8,600

What this gets you:
- 64 GB combined VRAM (no NVLink on 5090, so not unified)
- Strong local inference for 70B-class models (split across 2 GPUs)
- Local fine-tuning of 7B–13B models comfortably
- Heat: ~1.5 kW draw under load. You need to think about your room.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| 2-GPU build plan | Spec a complete build for your budget and target workloads; defend each choice | ⭐⭐ |
| Build and benchmark | Actually build a 2-GPU workstation; report `nccl-tests` numbers | ⭐⭐⭐ |
| Power and thermals | Undervolt your GPUs; measure perf-per-watt change; design airflow | ⭐⭐⭐ |
| Jetson deployment | Deploy a quantized 7B model on Jetson Orin Nano; report tokens/sec | ⭐⭐⭐ |
| FPGA inference | Implement a small CNN on a Pynq or DE10-Nano board | ⭐⭐⭐⭐⭐ |
| Tenstorrent kernel | Write a non-trivial program on a Tenstorrent card via tt-metal | ⭐⭐⭐⭐ |
| Tinytapeout submission | Submit a small digital design for a shared-wafer fab run | ⭐⭐⭐⭐⭐ |

### Key Insight

There is a meaningful distinction between **"DIY AI hardware"** (the actual silicon — mostly not feasible for individuals) and **"DIY AI systems"** (everything around the silicon — extremely feasible and high-leverage). Most of what makes a frontier-lab training run go fast is *systems work*: cooling, power, network, scheduling, custom kernels, quantization, serving. An individual or small team can absolutely contribute at the kernel and systems layer — that's exactly where many recent open-source wins (FlashAttention, vLLM, llama.cpp, Mojo, Triton kernels in HF) came from. Pick the layer where you can have impact and resist the urge to fight battles at the silicon layer.

### Resources

- [Tinytapeout](https://tinytapeout.com/) — open-source ASIC for hobbyists; small but real
- [Tenstorrent tt-metal](https://github.com/tenstorrent/tt-metal) — open kernel SDK
- [PYNQ](https://www.pynq.io/) — Python + FPGA; the friendliest FPGA on-ramp
- [Xilinx/AMD Alveo](https://www.amd.com/en/products/accelerators/alveo.html) — datacenter FPGAs
- [Skywater130 PDK](https://github.com/google/skywater-pdk) — open-source process design kit
- [Subreddits: r/LocalLLaMA, r/MachineLearning](https://www.reddit.com/r/LocalLLaMA/) — practical multi-GPU build wisdom

---

## Phase 10: Frontier Topics

Where AI hardware is going. Pick one or two threads.

### Co-Packaged Optics
Optical interconnects integrated directly into the GPU package, replacing copper for in-rack links. Solves the bandwidth-per-watt wall coming for electrical interconnects. NVIDIA, Broadcom, and TSMC all working on this; first production shipments expected 2026–2027.

### Chiplet Architectures
Building GPUs from multiple smaller dies connected by ultra-high-bandwidth interfaces (NVIDIA Blackwell uses two reticle-limited dies as one chip; AMD MI300 is built from many chiplets). Allows scaling past the reticle limit of monolithic dies. Will continue.

### Compute-in-Memory and Processing-Near-Memory
Pushing compute onto the HBM stack itself. Samsung's HBM-PIM, SK Hynix AiM. Promising for memory-bound workloads (read: most of deep learning). Real, but slow to commercialize.

### Analog AI
Use physical processes (resistance, photonics, capacitance) to do matmul in continuous-valued hardware. Mythic, Lightmatter, Lightelligence, Rain AI. Compelling theoretical efficiency; integration with digital systems is the hard part.

### Neuromorphic Computing
Spiking neural networks on event-driven hardware. Intel Loihi 2, IBM NorthPole. Niche but persistent; the gap to mainstream AI workloads is still large.

### LLM-Specific Hardware
Hardware designed specifically for transformer inference (Etched Sohu, Groq LPU). The bet: transformers are the workload of the decade, so build for them and abandon flexibility. The risk: architecture-of-the-future shifts and your ASIC is obsolete.

### Custom Silicon at Hyperscalers
Google (TPU), Amazon (Trainium/Inferentia), Meta (MTIA), Microsoft (Maia), OpenAI (rumored). Every major AI buyer is now also an AI chip designer. The economics: even a partial reduction in unit cost at hyperscaler volumes pays for the chip program many times over.

### Sustainability and Power
Power and cooling, not silicon, are increasingly the binding constraint on AI buildout. 1 GW datacenter campuses are now routine. Direct liquid cooling (DLC), immersion cooling, and even more exotic approaches are becoming standard. The energy story is becoming an AI story.

### Resources for the Frontier
- [SemiAnalysis](https://www.semianalysis.com/) — paywalled but the best ongoing analysis
- [Chips and Cheese](https://chipsandcheese.com/) — accessible silicon deep dives
- [HotChips conference](https://hotchips.org/) — annual conference; all the architectures get unveiled here
- [GPU MODE community](https://github.com/gpu-mode/lectures) — practical and current

---

## Suggested Timeline

| Phase | Duration | Outcome |
|-------|----------|---------|
| 0. Prerequisites | 0–2 weeks | C/C++/PyTorch comfortable; profilers installed |
| 1. Compute fundamentals | 1 week | Can compute and reason about roofline + arithmetic intensity |
| 2. GPU architecture | 1–2 weeks | Have a working mental model of an SM and warps |
| 3. Memory hierarchy | 1–2 weeks | Profile reveals exactly which level of memory you're hitting |
| 4. CUDA + Triton | 3–4 weeks | Shipped a custom kernel that beats eager PyTorch |
| 5. Other accelerators | 1–2 weeks | Ran inference on at least 2 non-NVIDIA hardware types |
| 6. Interconnects | 1–2 weeks | Trained on multi-node; understand NCCL collectives |
| 7. Quantization | 2 weeks | Quantized a 7B model with measurable quality understanding |
| 8. Inference systems | 1–2 weeks | Deployed a model with vLLM, understand prefill/decode |
| 9. DIY | 1–4 weeks | Specced or built a multi-GPU rig, or shipped a Jetson deployment |
| 10. Frontier | Ongoing | Tracking one thread (CIM, optics, chiplets, hyperscaler silicon) |

**Total to "comfortable hardware-aware practitioner":** ~3 months focused, longer if you want to be kernel-level fluent.

---

## Key Advice

1. **Profile before optimizing.** You will be wrong about where the time is going. Always. Use `nsys` and `ncu`, not `time.time()`.
2. **Arithmetic intensity is the single most important concept.** If you internalize one thing from this guide, make it the roofline.
3. **Memory-bound is the default.** Most deep-learning kernels are memory-bound, not compute-bound. Optimize for bandwidth and fusion, not FLOPs.
4. **`bf16` for training, `fp8`/`int4` for inference, `fp32` only when you must.** Mostly transparent on modern hardware.
5. **Always sync before timing CUDA.** CUDA is asynchronous; naive `time.time()` measures kernel launches, not kernel runs.
6. **HBM capacity is the constraint, not FLOPs.** A 7B model fits comfortably; a 70B doesn't. Choose hardware for memory first.
7. **NVLink ≫ PCIe ≫ network.** Plan your multi-GPU strategy around the link bandwidths.
8. **Quantization is mostly free for inference.** INT4 weight-only quantization typically loses 1–2% on benchmarks. Use it.
9. **Don't try to compete with cuBLAS on matmul.** You will lose. Write custom kernels for what cuBLAS can't or won't do.
10. **Watch the inference economics.** Training is one-time; inference is forever. Hardware for serving has different optimal points than hardware for training.

---

## Common Pitfalls to Avoid

- ❌ Timing CUDA without `torch.cuda.synchronize()` and getting absurd numbers
- ❌ Optimizing the wrong kernel because you didn't profile
- ❌ Trying to outpace cuBLAS on standard matmul
- ❌ Buying an A100 when you needed memory and an H100 was cheaper per HBM-byte
- ❌ Using FP16 when BF16 would have been free and stable
- ❌ Building a 4-GPU rig with a CPU that only exposes 16 PCIe lanes
- ❌ Quantizing aggressively for training (FP8 sometimes works; FP4/INT8 generally don't)
- ❌ Believing spec-sheet TFLOPs as if real workloads achieve them
- ❌ Designing distributed training without checking your NCCL topology first
- ❌ Trying to "DIY a GPU" or "design my own NPU" as an individual project

---

## Additional Resources

### Books and Long-Form Reading
- [CS:APP — Computer Systems: A Programmer's Perspective](https://csapp.cs.cmu.edu/)
- [Hennessy & Patterson — Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1)
- [Programming Massively Parallel Processors (Kirk & Hwu)](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) — the CUDA bible
- [Patterson & Hennessy — Computer Organization and Design](https://www.elsevier.com/books/computer-organization-and-design-risc-v-edition/patterson/978-0-12-820331-6) — for the systems gap

### Key Papers and Whitepapers
| Topic | Reference |
|-------|-----------|
| Roofline model | [Williams et al., 2009](https://dl.acm.org/doi/10.1145/1498765.1498785) |
| Original TPU | [Jouppi et al., 2017](https://arxiv.org/abs/1704.04760) |
| TPU v4 | [Jouppi et al., 2023](https://arxiv.org/abs/2304.01433) |
| H100 architecture | [NVIDIA H100 whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) |
| Blackwell architecture | [NVIDIA Blackwell whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture) |
| FlashAttention | [Dao et al., 2022](https://arxiv.org/abs/2205.14135) |
| vLLM / PagedAttention | [Kwon et al., 2023](https://arxiv.org/abs/2309.06180) |
| GPTQ | [Frantar et al., 2022](https://arxiv.org/abs/2210.17323) |
| QLoRA | [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314) |
| Speculative decoding | [Leviathan et al., 2022](https://arxiv.org/abs/2211.17192) |

### Tools You Should Know
- **`nsys` (Nsight Systems)** — timeline profiler
- **`ncu` (Nsight Compute)** — kernel-level profiler
- **`nvidia-smi`** + `nvidia-smi topo -m` — basic introspection
- **Triton** — kernel writing
- **PyTorch profiler + Perfetto** — application-level profiling
- **vLLM / SGLang / TensorRT-LLM** — serving stacks
- **`llama.cpp`** — cross-platform inference
- **CUTLASS** — matmul template library

### Communities
- [GPU MODE Discord](https://github.com/gpu-mode/lectures) — kernel-level work; the most active modern community
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) — practical hardware wisdom; multi-GPU builds, quantization
- [r/hardware](https://www.reddit.com/r/hardware/) — broader silicon discussion
- [HackerNews](https://news.ycombinator.com/) — most chip announcements get serious discussion
- Twitter/X — silicon analysts (Dylan Patel, Sravan Kundojjala, etc.) post regularly

---

## Quick Start Checklist

- [ ] Can sketch the roofline model and explain arithmetic intensity
- [ ] Can compute the AI for matmul, layernorm, softmax, and attention
- [ ] Know your GPU's compute capability, SM count, HBM size, and HBM bandwidth
- [ ] Have profiled a training step with `nsys` and identified the top kernels
- [ ] Have written a custom Triton kernel that beats eager PyTorch
- [ ] Can explain the difference between memory-bound and compute-bound
- [ ] Understand why decode is so much slower per FLOP than prefill
- [ ] Have quantized at least one model to INT4 and measured quality
- [ ] Have run NCCL benchmarks on a multi-GPU setup
- [ ] Know your NVLink topology via `nvidia-smi topo -m`
- [ ] Can spec a coherent multi-GPU workstation for a given budget
- [ ] Have served at least one model with vLLM and measured throughput
- [ ] Understand the difference between "DIY silicon" (mostly not) and "DIY systems" (yes)

---

## Glossary

| Term | Definition |
|------|------------|
| **AI (arithmetic intensity)** | FLOPs per byte of memory accessed; determines roofline position |
| **BF16** | Brain Floating Point 16; same range as FP32, less precision; modern training default |
| **CDNA / RDNA** | AMD's datacenter / consumer GPU architectures |
| **CUTLASS** | NVIDIA's open template library for matmul kernels |
| **FP8** | 8-bit floating point; E4M3 or E5M2 variants; modern inference default |
| **HBM** | High-Bandwidth Memory; stacked DRAM with TSV interconnects |
| **InfiniBand (IB)** | High-speed network with RDMA; standard for AI clusters |
| **KV cache** | Cached keys/values from prior tokens; grows linearly with context |
| **NCCL** | NVIDIA Collective Communications Library |
| **NVLink** | NVIDIA's GPU-GPU interconnect; much faster than PCIe |
| **NVSwitch** | NVLink switch chip; full-bandwidth all-to-all within a node |
| **PagedAttention** | Page-based KV cache management (vLLM) |
| **PCIe** | The standard CPU-GPU connection (and slower GPU-GPU when no NVLink) |
| **PTQ / QAT** | Post-Training / Quantization-Aware Training |
| **Roofline** | Performance model: min(peak FLOPs, BW × AI) |
| **SIMT** | Single Instruction Multiple Threads; NVIDIA's execution model |
| **SM** | Streaming Multiprocessor; the GPU's "core" |
| **Systolic array** | Data-flow matmul fabric used in TPUs |
| **Tensor Core** | Specialized matmul unit in NVIDIA GPUs since Volta |
| **TFLOPs** | Tera (10¹²) floating-point operations per second |
| **Triton** | Python-flavored GPU kernel language |
| **Warp** | 32 threads scheduled in lockstep on NVIDIA GPUs |

---

## License

MIT License. See the [LICENSE](../../LICENSE) file for details.
