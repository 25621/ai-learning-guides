# AI Learning Guides

A collection of long-form, project-driven guides for learning modern AI from first principles. Each guide takes a single topic from "I've heard of this" to "I can build it and debug it," organized into phases with runnable projects at every step.

> Built and maintained by a solo developer as personal learning notes — published in case they're useful to others. Code is runnable; expect rough edges.

---

## Guides

| Guide | Topic | Phases | Best for |
|---|---|---|---|
| [PyTorch Deep Dive](./guides/pytorch-deep-dive/) | Tensors, autograd, performance, distributed, custom kernels | 11 | Going from PyTorch user to power user |
| [LLM](./guides/llm/) | Transformers, GPT from scratch, pretraining, post-training, serving | 11 | Understanding and building language models |
| [Image Generation](./guides/image-generation/) | Autoencoders, GANs, diffusion | 6 | Generative vision from foundations to diffusion |
| [Reinforcement Learning](./guides/reinforcement-learning/) | MDPs, DQN, PPO, SAC, offline RL, RLHF | 11 | Learning RL as an algorithm family |
| [Video Generation](./guides/video-generation/) | Video diffusion, latent video, DiTs, world models | 11 | Temporal generative models |
| [Robotics](./guides/robotics/) | Control, perception, imitation learning, diffusion policies, VLAs, sim-to-real | 11 | Building robot learning systems |
| [Multimodal Learning](./guides/multimodal-learning/) | CLIP, fusion, VLMs, any-to-any models | 11 | Combining modalities into shared representations |
| [AI Hardware](./guides/ai-hardware/) | GPU architecture, CUDA/Triton, quantization, serving | 11 | Making models fast on real silicon |

The order above is a suggested learning progression: foundations first, then single-modality work, then cross-modality and applied tracks, with the systems-heavy AI Hardware guide as an orthogonal endpoint. Each guide is self-contained but cross-references the others where it makes sense. Project folders live under each guide's `projects/` directory — see [the structure section](#repository-structure) below.

---

## Where to start

There is no single "first" guide. Where to start depends on what you're trying to build. A few common paths:

### "I'm comfortable with deep learning basics, I want to actually understand what I'm using"
→ **[PyTorch Deep Dive](./guides/pytorch-deep-dive/)**. This is the foundation. If `view` vs `reshape`, autograd internals, or `torch.compile` feel hand-wavy to you, fix that first. Nearly every other guide assumes this fluency.

### "I want to build language models"
→ **[LLM](./guides/llm/)**, with **[PyTorch Deep Dive](./guides/pytorch-deep-dive/)** as a parallel reference. Hit Phase 6 (post-training / RLHF) and you'll want the RLHF section of the **[Reinforcement Learning](./guides/reinforcement-learning/)** guide too.

### "I want to build agents / robots that learn"
→ **[Reinforcement Learning](./guides/reinforcement-learning/)** for the algorithms, then **[Robotics](./guides/robotics/)** for applying them to physical systems. Robotics also leans on imitation learning and diffusion policies, which don't require RL — you can read robotics first and dip into RL on demand.

### "I want to build generative models for vision"
→ **[Image Generation](./guides/image-generation/)** first (autoencoders → GANs → diffusion). Then **[Video Generation](./guides/video-generation/)**, which assumes image-side diffusion fluency. **[Multimodal Learning](./guides/multimodal-learning/)** is the natural next step if you want text↔image↔video.

### "I want to make models actually fast"
→ **[AI Hardware](./guides/ai-hardware/)**. Different prerequisite stack from the others — systems-programming chops matter more than ML depth. Pair with **[PyTorch Deep Dive](./guides/pytorch-deep-dive/)** Phases 5–6 (performance, custom kernels).

### "I'm starting from scratch"
The honest answer: pick a project you actually want to build and work backward. Generic "learn AI" curricula tend to stall. If you want a default, do **[PyTorch Deep Dive](./guides/pytorch-deep-dive/)** → **[LLM](./guides/llm/)** Phases 1–3 → pick a direction.

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
       └──────────────┬───────────────────────────────┘
                      ▼
              ┌────────────────┐
              │  Multimodal    │  ← combines modalities;
              │   Learning     │     read after LLM + Image Gen
              └────────────────┘

                AI Hardware  ── orthogonal; pair with PyTorch Deep Dive
                               (systems-programming track, not ML track)
```

**Hard dependencies:**
- Video Generation assumes Image Generation (diffusion, U-Net, latent diffusion).
- Robotics borrows continuous-control RL from the RL guide (SAC, PPO, sim-to-real).
- Multimodal Learning assumes you've seen both a transformer-based LLM and an image encoder.

**Soft dependencies:**
- LLM Phase 6 (RLHF) reads better after RL Phase 9.
- Robotics Phase 6 (VLAs) reads better after Multimodal Phase 5 (VLMs).

---

## Repository structure

```
ai-learning-guides/
├── README.md                     ← you are here
├── shared/
│   ├── glossary.md               ← terms used across multiple guides
│   └── prerequisites.md          ← common Phase 0 material
├── guides/
│   ├── llm/
│   │   ├── README.md             ← the guide itself
│   │   ├── projects/
│   │   │   ├── 01-bigram-language-model/
│   │   │   │   ├── README.md     ← project explanation
│   │   │   │   ├── run.py        ← entry point
│   │   │   │   └── ...
│   │   │   ├── 02-tokenizer-bpe/
│   │   │   └── ...
│   │   ├── requirements.txt
│   │   └── STATUS.md             ← last-tested dates per project
│   ├── reinforcement-learning/
│   │   └── ... (same shape)
│   ├── pytorch-deep-dive/
│   ├── robotics/
│   ├── multimodal-learning/
│   ├── image-generation/
│   ├── video-generation/
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
| Generative vision track | PyTorch Deep Dive → Image Gen → Video Gen | 3 – 5 months |
| Robotics ML track | PyTorch Deep Dive → RL Phases 1–5 → Robotics | 4 – 6 months |
| Multimodal track | PyTorch Deep Dive → LLM Phases 1–3 → Image Gen Phases 1–5 → Multimodal | 5 – 7 months |
| Systems track | PyTorch Deep Dive → AI Hardware | 2 – 4 months |
| The whole thing | All eight guides, in dependency order | 12 – 18 months |

If a phase is taking 3x as long as the guide's suggested timeline, that's a signal — usually a prerequisite is shaky, not that you're slow.

---

## License

This guide is provided for educational purposes. Feel free to share and adapt.
