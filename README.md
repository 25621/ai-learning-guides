# AI Learning Guides

A collection of long-form, project-driven guides for learning modern AI from first principles. Each guide takes a single topic from "I've heard of this" to "I can build it and debug it," organized into phases with runnable projects at every step.

> Built and maintained by a solo developer as personal learning notes — published in case they're useful to others. Code is runnable; expect rough edges.

---

## Guides

| Guide | Topic | Phases | Best for |
|---|---|---|---|
| [PyTorch Deep Dive](./guides/pytorch-deep-dive/) | Tensors, autograd, performance, distributed, custom kernels | 11 | Going from PyTorch user to power user |
| [LLM](./guides/llm/) | Transformers, GPT from scratch, pretraining, post-training, serving | 11 | Understanding and building language models |
| [Image Generation](./guides/image-generation/) | Autoencoders, VAEs, GANs, diffusion, latent diffusion, DiTs, flow matching | 11 | Generative vision from foundations to frontier |
| [Reinforcement Learning](./guides/reinforcement-learning/) | MDPs, DQN, PPO, SAC, offline RL, RLHF | 11 | Learning RL as an algorithm family |
| [Video Generation](./guides/video-generation/) | Video diffusion, latent video, DiTs, world models | 11 | Temporal generative models |
| [Robotics](./guides/robotics/) | Control, perception, imitation learning, diffusion policies, VLAs, sim-to-real | 11 | Building robot learning systems |
| [Multimodal Learning](./guides/multimodal-learning/) | CLIP, fusion, VLMs, any-to-any models | 11 | Combining modalities into shared representations |
| [Inference Systems](./guides/inference-systems/) | KV cache, batching, speculative decoding, quantization, distributed serving | 11 | Putting trained LLMs into production |
| [AI Hardware](./guides/ai-hardware/) | GPU architecture, CUDA/Triton, quantization, serving | 11 | Making models fast on real silicon |

The order above is a suggested learning progression: foundations first, then single-modality work, then cross-modality and applied tracks, with the systems-heavy guides (Inference Systems and AI Hardware) as the production-and-performance endpoint. Each guide is self-contained but cross-references the others where it makes sense. Project folders live under each guide's `projects/` directory — see [the structure section](#repository-structure) below.

---

## Where to start

There is no single "first" guide. Where to start depends on what you're trying to build. A few common paths:

### "I'm comfortable with deep learning basics, I want to actually understand what I'm using"
→ **[PyTorch Deep Dive](./guides/pytorch-deep-dive/)**. This is the foundation. If `view` vs `reshape`, autograd internals, or `torch.compile` feel hand-wavy to you, fix that first. Nearly every other guide assumes this fluency.

### "I want to build language models"
→ **[LLM](./guides/llm/)**, with **[PyTorch Deep Dive](./guides/pytorch-deep-dive/)** as a parallel reference. Hit Phase 6 (post-training / RLHF) and you'll want the RLHF section of the **[Reinforcement Learning](./guides/reinforcement-learning/)** guide too. When you're ready to put a trained model in front of users, continue into **[Inference Systems](./guides/inference-systems/)**.

### "I want to serve LLMs in production"
→ **[Inference Systems](./guides/inference-systems/)**. Assumes you've done at least Phases 1–3 of the **[LLM](./guides/llm/)** guide ("KV cache" and "decoder-only" should not feel fuzzy). Pair with **[AI Hardware](./guides/ai-hardware/)** Phases 4 + 7 when you hit kernel-level questions.

### "I want to build agents / robots that learn"
→ **[Reinforcement Learning](./guides/reinforcement-learning/)** for the algorithms, then **[Robotics](./guides/robotics/)** for applying them to physical systems. Robotics also leans on imitation learning and diffusion policies, which don't require RL — you can read robotics first and dip into RL on demand.

### "I want to build generative models for vision"
→ **[Image Generation](./guides/image-generation/)** first (autoencoders → GANs → diffusion). Then **[Video Generation](./guides/video-generation/)**, which assumes image-side diffusion fluency. **[Multimodal Learning](./guides/multimodal-learning/)** is the natural next step if you want text↔image↔video.

### "I want to make models actually fast"
→ **[AI Hardware](./guides/ai-hardware/)**. Different prerequisite stack from the others — systems-programming chops matter more than ML depth. Pair with **[PyTorch Deep Dive](./guides/pytorch-deep-dive/)** Phases 5–6 (performance, custom kernels).

### "I'm starting from scratch"
The honest answer: pick a project you actually want to build and work backward. Generic "learn AI" curricula tend to stall. If you want a default, do **[PyTorch Deep Dive](./guides/pytorch-deep-dive/)** → **[LLM](./guides/llm/)** Phases 1–3 → pick a direction.

---

## Prerequisites

These apply to every guide in this collection. Each guide's **Phase 0: Prerequisites** layers the topic-specific concepts and tooling on top of these.

### Concepts to Know

- **Python**: classes, decorators, context managers, generators, virtual environments
- **Linear algebra**: matrix multiplication, vector spaces, broadcasting, basic eigenvalue intuition
- **Calculus**: gradients, chain rule, partial derivatives
- **Probability**: random variables, expectation, conditional probability
- **Deep learning basics**: training loops, loss functions, backpropagation, what an `nn.Module` is. The [PyTorch Deep Dive](./guides/pytorch-deep-dive/) is the recommended foundation if any of this feels shaky.
- **Shell and git**: you will read and clone a lot of repos

### What You Need Installed

- **Python 3.10+**, NumPy, PyTorch
- **A GPU** — owned, rented, or borrowed. Cloud is fine. Each guide notes its specific VRAM and hardware needs.

### Resources

- [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) — the visual intuition
- [Goodfellow, Bengio, Courville — Deep Learning Book](https://www.deeplearningbook.org/) — the standard reference
- [PyTorch official docs](https://pytorch.org/docs/stable/index.html) — your most-used reference across guides
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — the cleanest on-ramp to modern deep learning

---

## Dependency graph

How the guides relate to each other:

```
                    ┌──────────────────────┐
                    │  PyTorch Deep Dive   │  ← foundation for everything
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
  ┌──────────┐          ┌────────────┐         ┌──────────────┐
  │   LLM    │          │     RL     │         │  Image Gen   │
  └────┬─────┘          └──────┬─────┘         └──────┬───────┘
       │                       │                      │
       │                       ▼                      ▼
       │                ┌────────────┐         ┌──────────────┐
       │                │  Robotics  │         │  Video Gen   │
       │                └────────────┘         └──────────────┘
       │                                              │
       ├──────────────────┬───────────────────────────┘
       │                  │
       ▼                  ▼
┌──────────────┐   ┌────────────────┐
│  Inference   │   │  Multimodal    │  ← combines modalities;
│   Systems    │   │   Learning     │     read after LLM + Image Gen
└──────────────┘   └────────────────┘
       │
       │  shares territory (kernels, quantization)
       ▼
┌──────────────┐
│ AI Hardware  │  ← orthogonal systems track;
└──────────────┘     pair with PyTorch Deep Dive
```

**Hard dependencies:**
- Video Generation assumes Image Generation (diffusion, U-Net, latent diffusion).
- Robotics borrows continuous-control RL from the RL guide (SAC, PPO, sim-to-real).
- Multimodal Learning assumes you've seen both a transformer-based LLM and an image encoder.
- Inference Systems assumes LLM Phases 1–3 (transformer mechanics, KV cache concept).

**Soft dependencies:**
- LLM Phase 6 (RLHF) reads better after RL Phase 9.
- Robotics Phase 6 (VLAs) reads better after Multimodal Phase 5 (VLMs).
- Inference Systems Phases 5–6 (quantization, kernels) overlap with AI Hardware Phases 4 and 7; read together if both interest you.

---

## Repository structure

```
ai-learning-guides/
├── README.md                     ← you are here (cross-guide Prerequisites + Glossary live in this file)
├── guides/
│   ├── pytorch-deep-dive/
│   │   ├── README.md             ← the guide itself
│   │   ├── projects/
│   │   │   ├── 01-stride-explorer/
│   │   │   │   ├── README.md     ← project explanation
│   │   │   │   ├── run.py        ← entry point
│   │   │   │   └── ...
│   │   │   ├── 02-micrograd-pytorch-style/
│   │   │   └── ...
│   │   ├── requirements.txt
│   │   └── STATUS.md             ← last-tested dates per project
│   ├── llm/
│   │   └── ... (same shape)
│   ├── image-generation/
│   │   └── ... (same shape)
│   ├── reinforcement-learning/
│   │   └── ... (same shape)
│   ├── video-generation/
│   ├── robotics/
│   ├── multimodal-learning/
│   ├── inference-systems/
│   └── ai-hardware/
└── LICENSE
```

**Conventions used across all guides:**

- **Phases.** Each guide is split into ~6–11 phases that go from foundations to frontier. Phase 0, when present, is prerequisites.
- **Projects.** Each phase ends with a projects table. Projects are difficulty-rated (⭐ to ⭐⭐⭐⭐⭐) and live as folders under `guides/<topic>/projects/`, numbered sequentially across the whole guide (not per phase).
- **Bidirectional links.** Every project README links back to its phase; every phase links forward to its projects.
- **Runnable, not tested.** Code is meant to run, but there's no CI. Each guide's `STATUS.md` notes when projects were last verified.
- **Pinned dependencies.** Each guide has its own `requirements.txt`. Versions are pinned loosely (`>=X,<Y`) — strict enough to avoid silent rot, loose enough not to break weekly.

---

## Project difficulty legend

| Tier | Meaning | Typical time |
|---|---|---|
| ⭐ | "Make sure you understand the concept" — single file, a few dozen lines | 30 min – 2 hours |
| ⭐⭐ | "Build a working version" — multiple files, real implementation | 2 – 8 hours |
| ⭐⭐⭐ | "Build something non-trivial" — design choices matter, debugging matters | 1 – 3 days |
| ⭐⭐⭐⭐ | "Reproduce a paper" — published result on a small scale; expect to fight your tooling | 1 – 2 weeks |
| ⭐⭐⭐⭐⭐ | "Research-level" — open-ended, likely no reference implementation matches yours exactly | 2+ weeks, often much more |

These are guidelines, not promises. ⭐⭐⭐⭐ projects routinely overrun their estimates; ⭐⭐⭐⭐⭐ projects are graded on what you learned, not whether they "worked."

---

## Philosophy

A few opinions that shape every guide here:

**Build first, derive later.** You learn faster when you have working code in front of you. Each phase opens with concepts, then code, then projects — in that order, but the projects are where it sticks.

**Explicit over magic.** Where there's a choice between calling a library and writing the loop, the guides write the loop at least once. You can use the library afterward with eyes open.

**Equations are checkpoints, not décor.** When an equation appears, it's because you'll need to recognize it in code five pages later. If you can't connect the math to the implementation, slow down.

**Frontier topics are flagged honestly.** The "Phase 10: Frontier" sections describe things actively being researched. They go stale fastest. Treat them as starting points, not conclusions.

**No prestige hierarchy of topics.** A working bigram model is more valuable than a half-built MoE. The ⭐ projects are not throwaway — they're the foundation the ⭐⭐⭐⭐⭐ ones rest on.

---

## Suggested learning timelines

These are rough — real time depends heavily on background and how much you build vs. read.

| Path | Guides | Approx. time (part-time) |
|---|---|---|
| LLM engineer track | PyTorch Deep Dive → LLM → RL Phase 9 (RLHF) | 4 – 6 months |
| LLM serving track | PyTorch Deep Dive → LLM Phases 1–3 → Inference Systems | 3 – 5 months |
| Generative vision track | PyTorch Deep Dive → Image Gen → Video Gen | 3 – 5 months |
| Robotics ML track | PyTorch Deep Dive → RL Phases 1–5 → Robotics | 4 – 6 months |
| Multimodal track | PyTorch Deep Dive → LLM Phases 1–3 → Image Gen Phases 1–5 → Multimodal | 5 – 7 months |
| Systems track | PyTorch Deep Dive → AI Hardware → Inference Systems | 3 – 5 months |
| The whole thing | All nine guides, in dependency order | 14 – 20 months |

If a phase is taking 3x as long as the guide's suggested timeline, that's a signal — usually a prerequisite is shaky, not that you're slow.

---

## License

MIT License. See the [LICENSE](https://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
