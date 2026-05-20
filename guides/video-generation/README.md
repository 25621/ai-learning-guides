# Video Generation: From Beginner to Advanced

A comprehensive guide to understanding and building video generation systems — from the fundamentals of treating video as a spatiotemporal signal, through latent video diffusion and Diffusion Transformers (DiT), to long-form generation, world models, and the frontier of real-time interactive video.

> **Video generation = image generation + time.** That one sentence is both true and dangerously misleading. The "+ time" introduces problems that have no image-gen analog: temporal consistency, motion priors, enormous compute (a 5-second 720p clip is ~150 images), and the brutal scarcity of high-quality paired video-text data. This guide is about how the field solved (and is still solving) those problems.

---

## Table of Contents

1. [Phase 0: Prerequisites](#phase-0-prerequisites)
2. [Phase 1: Foundations — Video as a Tensor](#phase-1-foundations--video-as-a-tensor)
3. [Phase 2: Classical and Early Neural Video Generation](#phase-2-classical-and-early-neural-video-generation)
4. [Phase 3: Image-to-Video as a Stepping Stone](#phase-3-image-to-video-as-a-stepping-stone)
5. [Phase 4: Video Diffusion — The Modern Foundation](#phase-4-video-diffusion--the-modern-foundation)
6. [Phase 5: Latent Video Diffusion and Video Tokenizers](#phase-5-latent-video-diffusion-and-video-tokenizers)
7. [Phase 6: Diffusion Transformers (DiT) and Sora-Class Models](#phase-6-diffusion-transformers-dit-and-sora-class-models)
8. [Phase 7: Conditioning, Control, and Editing](#phase-7-conditioning-control-and-editing)
9. [Phase 8: Long-Form and Consistent Video](#phase-8-long-form-and-consistent-video)
10. [Phase 9: World Models and Interactive Video](#phase-9-world-models-and-interactive-video)
11. [Phase 10: Training at Scale, Evaluation, and Frontier Topics](#phase-10-training-at-scale-evaluation-and-frontier-topics)
12. [Suggested Timeline](#suggested-timeline)
13. [Key Advice](#key-advice)
14. [Additional Resources](#additional-resources)
15. [Glossary](/shared/glossary/)

---

## Phase 0: Prerequisites

Video generation is one of the most demanding topics in modern ML. The prerequisites are non-negotiable.

### Concepts to Know

- **Diffusion models**: forward/reverse process, DDPM, DDIM, classifier-free guidance, noise schedules
- **Latent diffusion**: VAE encoder/decoder, training a diffusion model in latent space (i.e., Stable Diffusion)
- **U-Net architecture**: down/up blocks, skip connections, attention blocks
- **Transformers**: self-attention, cross-attention, positional embeddings, transformer blocks
- **Vision Transformers (ViT)**: patchification, 1D-sequence treatment of images
- **Text conditioning**: CLIP text encoders, T5, cross-attention for text→image
- **PyTorch fluency**: mixed precision, distributed training (DDP/FSDP), memory profiling
- **Optical flow** (helpful): what it is and why it shows up everywhere in video

### The One Equation Everything Comes Back To

```
A video is a tensor of shape (T, H, W, C) — frames × height × width × channels.

Modern video generation models a distribution over this tensor:
    p(x_video | text, image, audio, ...)

The dominant approach today: tokenize the video into a (T', H', W') latent
grid with a 3D VAE, then either
    (a) run diffusion in that latent space (Sora, Veo, MovieGen), or
    (b) autoregress next-token in that latent space (CogVideo, Phenaki),
    (c) or a hybrid.

The single hardest problem isn't the model — it's getting (T, H, W) all
big enough to be useful without compute exploding cubically.
```

### Resources

- [Image Generation guide](../image-generation/) — strongly recommended prerequisite
- [Lilian Weng — What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — the canonical primer
- [Sora Technical Report (OpenAI, 2024)](https://openai.com/index/video-generation-models-as-world-simulators/) — read once now, again at the end of Phase 6
- [Stable Video Diffusion paper](https://arxiv.org/abs/2311.15127) — practical entry point

---

## Phase 1: Foundations — Video as a Tensor

Before models, understand the data. Video has properties that images don't, and they shape every architectural decision later.

### Concepts to Learn

- **Video shape conventions**: `(B, T, C, H, W)` (PyTorch) vs `(B, C, T, H, W)` (3D conv-friendly) — both common, easy to confuse
- **Frame rate (fps)** — 24, 25, 30, 60; the same motion at different fps looks very different to a model
- **Video codecs**: H.264, H.265/HEVC, AV1, VP9 — most public video is heavily compressed; this matters
- **Color spaces**: YUV420 (native to most codecs) vs RGB (what your model wants)
- **Containers vs codecs**: `.mp4`, `.mov`, `.webm` are containers; H.264, AV1 are the codecs inside them
- **Temporal redundancy**: adjacent frames are nearly identical — both a problem (waste) and an opportunity (compression)
- **Motion as a signal**: optical flow, motion vectors (already inside the codec), scene cuts
- **Data loading is brutal**: a 1-minute 1080p clip is gigabytes uncompressed; decode-on-the-fly is mandatory

### The Cost of a Single Clip

```
Resolution × fps × duration → raw tensor size

  256×256, 8 fps, 2 sec  →  16 frames × 256 × 256 × 3 = 3.1 MB (one clip!)
  512×512, 24 fps, 5 sec → 120 frames × 512 × 512 × 3 = 94 MB
  720p,   24 fps, 5 sec  → 120 × 1280 × 720 × 3       = 333 MB
  1080p,  24 fps, 10 sec → 240 × 1920 × 1080 × 3      = 1.5 GB

(All in fp32; halve for fp16 / bf16.)

→ A batch size of 8 at 1080p × 10s is 12 GB just for inputs.
  This is why every video model uses a latent VAE.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Video loader benchmark | Compare `torchvision.io`, `decord`, `pyav`, and `ffmpeg-python` on a folder of `.mp4`s; report decode time per clip | ⭐⭐ |
| Frame extractor | Sample N frames evenly from a clip; sample N frames at uniform fps; observe the difference for fast vs slow scenes | ⭐⭐ |
| Optical flow visualizer | Compute dense optical flow (RAFT, Farnebäck) between adjacent frames; color-visualize | ⭐⭐ |
| Scene-cut detector | Detect scene boundaries via histogram or feature distance; split a movie into clips | ⭐⭐ |
| Storage study | Take 100 clips, store as raw `.npy`, H.264 `.mp4`, and AV1 `.webm`; compare disk and decode speed | ⭐⭐ |

### Sample Code: Loading a Video Clip with `decord`

```python
import decord
import torch
from decord import VideoReader

decord.bridge.set_bridge("torch")    # decode directly to torch tensors

vr = VideoReader("input.mp4", num_threads=2)
fps = vr.get_avg_fps()
total = len(vr)

# Sample 16 frames uniformly across the clip:
indices = torch.linspace(0, total - 1, 16).long().tolist()
frames = vr.get_batch(indices)         # (16, H, W, 3), uint8

# Convert to (T, C, H, W) float in [-1, 1] for model input:
frames = frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0
```

### Key Insight

Every operational decision in video generation — frame rate, clip length, resolution, batch size — is a compute-vs-quality trade-off, and they all multiply. Doubling resolution = 4× compute. Doubling frame count = 2× compute. Doubling batch size = 2× compute. Doubling all three = 16×. This is why the field obsesses over latent compression and why nearly every published video model lists its exact `(T, H, W)` operating point as a design parameter, not an afterthought.

### Resources

- [`decord`](https://github.com/dmlc/decord) — the standard fast video loader
- [`pyav`](https://pyav.org/) — Python bindings to ffmpeg
- [RAFT optical flow paper](https://arxiv.org/abs/2003.12039)
- [FFmpeg documentation](https://ffmpeg.org/documentation.html) — you will need it

---

## Phase 2: Classical and Early Neural Video Generation

The history matters — it's where you learn what doesn't work and why. Skim, don't memorize.

### Concepts to Learn

- **Frame interpolation** — generating intermediate frames between two real ones (FILM, Super SloMo); a "video generation lite"
- **Future frame prediction** — given a few frames, predict the next ones (early benchmark: Moving MNIST)
- **Video GANs**:
  - VGAN, TGAN — early attempts, low quality
  - MoCoGAN — disentangled motion and content
  - DVD-GAN — first plausible-quality short clips
  - StyleGAN-V — applied StyleGAN's latent space to video
- **Autoregressive pixel models**: VideoPixelNetwork, slow but principled
- **Recurrent approaches**: ConvLSTM, PredRNN — used widely before transformers won
- **The limits of these approaches**: short, low-resolution, no text conditioning, mode collapse for GANs

### Why These Mostly Stopped

```
Around 2022 the field made three near-simultaneous moves that made
older approaches obsolete:

  1. Diffusion proved itself on images (DDPM → Imagen, Stable Diffusion).
  2. Latent compression made it tractable for high resolution.
  3. Text-image pretraining produced strong text conditioning for free.

Video inherited all three. GAN-based and pure-recurrent video generation
have not seriously competed with diffusion since ~2023.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Moving MNIST predictor | Train a ConvLSTM to predict the next 10 frames given 10; classic baseline | ⭐⭐⭐ |
| FILM frame interpolation | Use a pretrained FILM to interpolate between two real frames; observe motion artifacts | ⭐⭐ |
| Tiny video GAN | Train a small video GAN on UCF-101 face crops — observe mode collapse firsthand | ⭐⭐⭐⭐ |
| Read MoCoGAN | Implement just the latent-disentanglement idea (content + motion latents) in a small VAE | ⭐⭐⭐ |

### Key Insight

The pre-diffusion era of video generation is a graveyard of clever ideas that didn't scale. Most of them — disentangled motion latents, hierarchical generation, two-stream architectures — have since reappeared as components inside diffusion-based systems. The ideas were right; the training framework was wrong.

### Resources

- [Video Prediction Beyond Mean Square Error (Mathieu et al., 2015)](https://arxiv.org/abs/1511.05440) — early classic
- [MoCoGAN paper](https://arxiv.org/abs/1707.04993)
- [PredRNN paper](https://arxiv.org/abs/1804.06300)
- [FILM frame interpolation paper](https://arxiv.org/abs/2202.04901)

---

## Phase 3: Image-to-Video as a Stepping Stone

Before generating video from scratch, generate video from an *image*. This is the conceptually simplest version of the problem and the most practical to start training on.

### Concepts to Learn

- **The image-to-video (I2V) task** — given one frame, produce a clip starting from it
- **Conditioning on a still image**: concatenate, cross-attend, or AdaLN modulation
- **Motion buckets / motion scores** — letting the user control "how much motion"
- **Camera control** — explicit camera trajectory as a side input (CameraCtrl, MotionCtrl)
- **The two main outputs of an I2V model**: short clips (2–5 sec) and animated stills (subtle motion, longer)
- **Stable Video Diffusion (SVD)** — the canonical open-weights I2V model; freezes a pretrained image latent diffusion model and adds temporal layers
- **AnimateDiff** — adds a "motion module" to any community Stable Diffusion checkpoint without retraining the base

### Why I2V Is Easier Than T2V

```
Text-to-video (T2V):    text  → video      (no anchor; must invent everything)
Image-to-video (I2V):   image → video      (first frame fixes appearance,
                                            model only models motion)
Video-to-video (V2V):   video → video      (style transfer / restyling)

I2V's training signal is also cheaper: any video is automatically a
training example — first frame is the condition, the rest is the target.
No paired text needed.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Run SVD inference | Generate 14-frame and 25-frame clips with Stable Video Diffusion from arbitrary images | ⭐⭐ |
| AnimateDiff tour | Plug AnimateDiff's motion module into a community SD 1.5 checkpoint; generate animated stills | ⭐⭐⭐ |
| Tiny I2V model | Add 3D temporal conv layers to a frozen SD 1.5 U-Net; fine-tune on 100k clips with the first frame as condition | ⭐⭐⭐⭐⭐ |
| Motion control | Train the above with a motion-score input; verify that low scores produce subtle motion | ⭐⭐⭐⭐ |
| Camera trajectory | Add Plücker-coordinate camera embeddings to an I2V model; verify pan/zoom controllability | ⭐⭐⭐⭐⭐ |

### Sample Code: Inflating a 2D Conv to a (2+1)D Conv

```python
import torch
import torch.nn as nn

class Conv2Plus1D(nn.Module):
    """Common pattern: factorize a 3D conv into spatial + temporal."""
    def __init__(self, in_c, out_c, k_s=3, k_t=3):
        super().__init__()
        self.spatial = nn.Conv3d(in_c, out_c, kernel_size=(1, k_s, k_s),
                                 padding=(0, k_s // 2, k_s // 2))
        self.temporal = nn.Conv3d(out_c, out_c, kernel_size=(k_t, 1, 1),
                                  padding=(k_t // 2, 0, 0))

    def forward(self, x):
        # x: (B, C, T, H, W)
        return self.temporal(self.spatial(x))

# Init temporal as identity (zeros + identity in middle) so a 2D-pretrained
# model passes video through unchanged at the start of training:
def init_temporal_as_identity(conv):
    nn.init.zeros_(conv.weight)
    middle = conv.kernel_size[0] // 2
    for c in range(min(conv.in_channels, conv.out_channels)):
        conv.weight.data[c, c, middle, 0, 0] = 1.0
```

### Key Insight

The dominant pattern across nearly all 2022–2024 video diffusion models is **temporal inflation**: take a pretrained image model, insert temporal layers initialized as identity, and fine-tune. This preserves the pretrained spatial knowledge while learning motion on top. AnimateDiff, ModelScope, Stable Video Diffusion, and Make-A-Video all use variants of this trick. The 2024–2026 frontier (Sora, Veo, Movie Gen) abandons it in favor of training spatiotemporal models from scratch — but the inflation pattern is still the right starting point for any custom model.

### Resources

- [Stable Video Diffusion paper](https://arxiv.org/abs/2311.15127)
- [AnimateDiff paper](https://arxiv.org/abs/2307.04725)
- [Make-A-Video paper](https://arxiv.org/abs/2209.14792)
- [CameraCtrl paper](https://arxiv.org/abs/2404.02101)

---

## Phase 4: Video Diffusion — The Modern Foundation

This is where the field is. Master this phase deeply; the next two are refinements.

### Concepts to Learn

- **Pixel-space vs latent-space video diffusion** — pixel space is impractical at any meaningful resolution; latent space is the default
- **3D U-Nets** — the natural generalization of 2D U-Nets to (T, H, W)
- **(2+1)D factorization** — separate spatial and temporal layers; cheaper and easier to initialize from 2D pretrained weights
- **Temporal attention** — pure attention along the time axis at each spatial position; the modern default for high-quality models
- **Spatiotemporal attention** — joint attention over `(T × H × W)`; quadratic in sequence length and very expensive
- **Video-CFG**: classifier-free guidance for video; balancing text alignment against temporal coherence
- **Cascaded diffusion** for video: low-res video → super-resolution → frame interpolation (Imagen Video, Make-A-Video used this; modern models do it less)
- **Noise schedules for video** — empirically need lower SNR (more noise) than images at the same resolution
- **Joint image-video training** — co-train on still images (treated as 1-frame video) to maintain image quality

### The 3D U-Net Block

```
Input: (B, C, T, H, W)
                                            
┌─────────────────────────────────────────┐ 
│ Spatial conv (1×3×3)         ──┐        │ inflated 2D conv
│ Spatial self-attention       ──┤        │ shared with image weights
│ Cross-attention (text)       ──┘        │
│                                          │
│ Temporal conv (3×1×1)        ──┐        │
│ Temporal self-attention      ──┤        │ new, initialized as identity
│                              ──┘        │
└─────────────────────────────────────────┘

Modern variant: replace all "conv" with "transformer block" → DiT (Phase 6).
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Inflate SD to a video model | Take a Stable Diffusion 1.5 U-Net, inflate to 3D (insert temporal conv + temporal attention), train on a small video dataset | ⭐⭐⭐⭐⭐ |
| Joint image-video training | Co-train your inflated model on 90% images, 10% video; compare to video-only training on quality and motion | ⭐⭐⭐⭐ |
| Temporal CFG study | Vary CFG strength independently for text and for image conditioning; observe trade-offs | ⭐⭐⭐ |
| Cascaded super-resolution | Build a small "low-res video → high-res video" diffusion super-resolution model | ⭐⭐⭐⭐ |
| Compare attention patterns | (2+1)D vs full spatiotemporal vs windowed spatiotemporal; measure FLOPs and quality | ⭐⭐⭐⭐ |

### Sample Code: A (2+1)D Transformer Block for Video

```python
import torch
import torch.nn as nn
from einops import rearrange

class Video2Plus1DBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_s = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        # Zero-init the temporal-attention output projection so the model
        # behaves as a still-image model at initialization:
        nn.init.zeros_(self.temporal_attn.out_proj.weight)
        nn.init.zeros_(self.temporal_attn.out_proj.bias)

    def forward(self, x):
        # x: (B, T, S, D) where S = H*W spatial positions
        B, T, S, D = x.shape

        # Spatial: each frame independently attends within itself
        h = rearrange(x, "b t s d -> (b t) s d")
        h_norm = self.norm_s(h)
        h = h + self.spatial_attn(h_norm, h_norm, h_norm, need_weights=False)[0]

        # Temporal: each spatial position attends across time
        h = rearrange(h, "(b t) s d -> (b s) t d", b=B, t=T)
        h_norm = self.norm_t(h)
        h = h + self.temporal_attn(h_norm, h_norm, h_norm, need_weights=False)[0]

        return rearrange(h, "(b s) t d -> b t s d", b=B, s=S)
```

### Key Insight

The single biggest design lever in video diffusion is **what gets attention along the time axis**. Pure (2+1)D — spatial attention then separate temporal attention — is cheap and works surprisingly well, which is why it dominated 2023–2024. Full spatiotemporal attention is much more expressive but quadratic in `T×H×W`, which gets prohibitive fast. Modern Sora-class models pay this cost using 3D latent compression to shrink `T×H×W` aggressively before attention runs. The trick is moving the expense from attention into the VAE.

### Resources

- [Video Diffusion Models (Ho et al., 2022)](https://arxiv.org/abs/2204.03458) — first principled paper
- [Imagen Video paper](https://arxiv.org/abs/2210.02303) — cascaded approach, lots of useful detail
- [Make-A-Video paper](https://arxiv.org/abs/2209.14792)
- [Align Your Latents (Blattmann et al., 2023)](https://arxiv.org/abs/2304.08818) — the latent video diffusion paper

---

## Phase 5: Latent Video Diffusion and Video Tokenizers

The single most important enabler of modern video generation. If you understand the VAE in image-gen, this is the natural extension — but the engineering is much harder.

### Concepts to Learn

- **Why latent space is non-negotiable** — see Phase 1's storage table
- **2D VAEs for video** — run a 2D image VAE per frame; works, but no temporal compression
- **3D VAEs for video** — compress in time as well as space; the modern default. Typical compression ratios: 4× temporal, 8× spatial → 32–128× total
- **Causal 3D VAEs** — first frame encoded with itself only, later frames encoded with causal context. Lets the same model handle still images and video
- **Reconstruction quality matters more than for images** — temporal flicker in the VAE shows up directly as motion artifacts
- **Discrete vs continuous latents**:
  - **Continuous** (VAE) for diffusion models
  - **Discrete** (VQ-VAE, FSQ, LFQ) for autoregressive/transformer-style models — MagViT-v2 is the strongest open recipe
- **Joint training with images** — same caveat as the U-Net case; helps preserve still-image quality
- **Two-stage training**: train the VAE first, freeze it, then train the diffusion model in its latent space

### Latent Compression in Numbers

```
Raw clip:           120 frames × 720p × 3 channels = 333 MB

After 3D VAE:
  Spatial:  8×8 compression  → 90×128 per frame
  Temporal: 4× compression   → 30 frames
  Channels: e.g., 16         → (30, 90, 128, 16) ≈ 21 MB

That's 16× less data — and crucially, diffusion now runs over 30 latent
"frames" instead of 120. Memory and compute both drop dramatically.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Frame-by-frame 2D VAE | Use Stable Diffusion's VAE on video frames independently; observe temporal flicker in reconstructions | ⭐⭐ |
| Train a small 3D VAE | (B, 3, T, H, W) → (B, C, T', H', W'); compress 4× in time, 8× in space; train on UCF-101 | ⭐⭐⭐⭐⭐ |
| Causal 3D VAE | Modify the above to causal in time so it handles single images correctly (T=1 → T'=1) | ⭐⭐⭐⭐ |
| MagViT-v2-style tokenizer | Train a discrete video tokenizer using FSQ or LFQ quantization; measure reconstruction FID | ⭐⭐⭐⭐⭐ |
| Diffusion on latents | Plug the 3D VAE in front of a small diffusion model from Phase 4; compare training speed and quality to pixel-space | ⭐⭐⭐⭐ |

### Key Insight

The 3D VAE is the unsung hero of modern video generation. Sora's "patches" — its much-discussed innovation — are just patches in the latent space of a 3D VAE. The trick is that a powerful enough VAE compresses video by ~100× while preserving the information that matters for generation, so the diffusion model can train on what used to be a 100-GB clip as if it were a 1-GB clip. Every "Sora-class" model has spent serious effort on its VAE; the best ones have spent at least as much effort there as on the diffusion backbone.

### Resources

- [MagViT-v2 paper (Yu et al., 2023)](https://arxiv.org/abs/2310.05737) — the strongest open recipe for video tokenization
- [OpenSora's VAE](https://github.com/hpcaitech/Open-Sora/blob/main/docs/vae.md) — open implementation
- [CogVideoX paper](https://arxiv.org/abs/2408.06072) — recent strong open VAE + DiT
- [VideoGPT paper](https://arxiv.org/abs/2104.10157) — original discrete-token approach

---

## Phase 6: Diffusion Transformers (DiT) and Sora-Class Models

The current frontier. As of 2026, the strongest video models are all DiT-based, trained on latent video tokens, with text conditioning via cross-attention or token concatenation.

### Concepts to Learn

- **DiT (Diffusion Transformer)** — Peebles & Xie's paper that replaced the U-Net with a pure transformer for image diffusion; the foundation
- **Patchification of latent video** — take the 3D-VAE latents `(T', H', W', C)`, patchify into a 1D sequence of spatiotemporal tokens
- **AdaLN-Zero** — the modulation scheme that DiT uses for conditioning; surprisingly robust
- **3D RoPE (Rotary Position Embedding)** — extends 2D RoPE to time; the standard now
- **Sora's "patches" design** — patches at *variable* size, allowing flexible resolution and aspect ratio at inference
- **Rectified Flow / Flow Matching** — modern replacement for DDPM training that's better-behaved at scale (used by SD3, Flux, and most 2024+ video models)
- **MMDiT (Multi-Modal DiT)** — the SD3 architecture: text and image tokens share attention layers; extended to video in Movie Gen and similar
- **Open-source replication efforts**:
  - **OpenSora** (HPC-AI Tech) — full open Sora replica
  - **Open-Sora-Plan** (PKU) — independent replica
  - **CogVideoX** (THUDM) — Tsinghua's strong open release
  - **HunyuanVideo** (Tencent) — large-scale open release
  - **Mochi 1** (Genmo) — high-quality open
  - **Wan2.1** (Alibaba) — recent strong open release
- **Closed frontier**:
  - **Sora / Sora-Turbo** (OpenAI)
  - **Veo / Veo 2** (Google DeepMind)
  - **Movie Gen** (Meta)
  - **Kling, Hailuo, Runway Gen-3, Luma Dream Machine** — commercial offerings

### Sora-Style Architecture, Sketched

```
Text prompt ──► T5 / CLIP text encoder ──► text tokens

Video latent: (T'=30, H'=90, W'=128, C=16) from 3D VAE
   │
   ▼ patchify (e.g., 2×2×2 patches)
Patches: 15 × 45 × 64 = 43,200 video tokens, each of dim P²×P×C×P → projected to D
   │
   ▼ concat or cross-attend with text tokens
   │
   ▼ MANY transformer blocks (e.g., 40–80 blocks, hidden dim 1500+)
   │  each with: 3D-RoPE, self-attention over all video+text tokens,
   │             AdaLN modulation from timestep & conditioning, MLP
   │
   ▼ predict noise (or velocity, for rectified flow)
   │
   ▼ DDPM/flow-matching loss on the (B, N_tokens, D) prediction

At inference:
   start from Gaussian noise in latent space
   denoise over ~30–50 steps (flow matching with few steps)
   un-patchify, decode through 3D VAE → pixel video
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Implement DiT for video | Take a published DiT image implementation; extend to (T, H, W) patches and 3D RoPE; train on a small video dataset | ⭐⭐⭐⭐⭐ |
| Flow matching from scratch | Replace DDPM with rectified flow / flow matching in a small video DiT; compare convergence | ⭐⭐⭐⭐ |
| Read and reproduce OpenSora | Run inference on a pretrained OpenSora checkpoint; modify one component (e.g., the VAE), retrain | ⭐⭐⭐⭐⭐ |
| MMDiT for video | Implement the SD3-style joint text-video attention; verify text adherence improves | ⭐⭐⭐⭐⭐ |
| Variable resolution | Modify your DiT to handle arbitrary `(T, H, W)` at inference (Sora's claim); test on aspect ratios it didn't see at training | ⭐⭐⭐⭐⭐ |

### Key Insight

The shift from U-Net to DiT in image gen took ~18 months to play out fully (~2022→2024). The same shift in video gen is happening *now*, faster, because the lesson has been learned. Any video model started in 2025 onward almost certainly uses a transformer backbone, latent input, and flow matching. If you're learning the field for the first time, you can largely skip U-Net-based video models — they're being deprecated in real time. Understand them historically; build on DiT.

### Resources

- [DiT paper (Peebles & Xie, 2022)](https://arxiv.org/abs/2212.09748) — foundational
- [Sora technical report](https://openai.com/index/video-generation-models-as-world-simulators/)
- [Movie Gen technical report (Meta, 2024)](https://ai.meta.com/research/movie-gen/)
- [CogVideoX paper](https://arxiv.org/abs/2408.06072)
- [HunyuanVideo paper](https://arxiv.org/abs/2412.03603)
- [Rectified Flow paper (Liu et al., 2022)](https://arxiv.org/abs/2209.03003)
- [Flow Matching for Generative Modeling (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747)
- [OpenSora](https://github.com/hpcaitech/Open-Sora) and [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan)

---

## Phase 7: Conditioning, Control, and Editing

Generating a video is one thing; generating *the video you want* is another. This phase is about everything that wraps the core model.

### Concepts to Learn

- **Text conditioning** — T5 vs CLIP vs both (Imagen/SD3-style "use two encoders"); long-prompt handling
- **Image conditioning** — first-frame conditioning (I2V), last-frame, both, keyframes
- **Video-to-video** — restyling, depth-conditioned, pose-conditioned (ControlNet-Video)
- **Camera control** — explicit camera pose embeddings (Plücker coordinates) or motion-bucket conditioning
- **Motion control** — bounding-box trajectories, sparse motion strokes, dense motion maps
- **Identity preservation** — keeping a specific character or object consistent (DreamBooth-Video, ID-Animator)
- **Audio-conditioned video** — talking-head models (SadTalker, EMO, V-Express, Hallo), sync to lip motion
- **Video editing**:
  - **Inversion-based editing** — invert the video into latent noise, edit, denoise
  - **Token Merging for Video** — runtime acceleration
  - **Rerender / TokenFlow** — style transfer with temporal consistency
- **Negative prompts** for video — what unwanted artifacts you can subtract

### A Taxonomy of Video Generation Tasks

```
INPUT                     →  TASK                    EXAMPLES
─────────────────────────    ────────────────────    ────────────────────────
text                      →  T2V                     Sora, Veo, Kling
text + image              →  T+I2V (frame-locked)    SVD-XT, Kling I2V
text + first+last frame   →  keyframe interpolation  Frame Genie, Wan-FLF2V
image                     →  I2V (motion only)       SVD, animated stills
video + text              →  V2V restyle             Rerender, TokenFlow
video + pose/depth        →  controlled V2V          AnimateAnyone,
                                                     ControlNet-Video
audio + image             →  talking head            EMO, Hallo, V-Express
text + camera trajectory  →  cinematic T2V           MotionCtrl, CameraCtrl
text + object trajectory  →  trajectory-controlled   Boximator, DragAnything
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Long-prompt handling | Train or fine-tune with T5-XXL prompts (up to 256 tokens); compare against CLIP-L conditioning on adherence | ⭐⭐⭐⭐ |
| ControlNet-Video | Adapt ControlNet to a video diffusion model; condition on depth maps across all frames | ⭐⭐⭐⭐ |
| Camera control | Add Plücker-coordinate camera embeddings; verify pan / zoom / orbit work | ⭐⭐⭐⭐ |
| Talking head | Run EMO or Hallo on a portrait + audio clip; fine-tune for a specific speaker | ⭐⭐⭐⭐ |
| Video inversion + edit | Invert a real clip into latent noise; replace an object via prompt edit | ⭐⭐⭐⭐⭐ |
| LoRA for video | Train a video LoRA on ~50 clips of a specific style or character | ⭐⭐⭐⭐ |

### Key Insight

In image generation, ControlNet and its successors made the difference between "generate something cool" and "generate exactly what I want." Video is following the same trajectory but several years behind. The 2026 frontier in video isn't just bigger models — it's *better control surfaces*: camera trajectories, character consistency across cuts, dialogue lip sync, scene-level keyframe control. Whoever ships the "ControlNet moment" for video at the right level of abstraction defines the next generation of commercial tools.

### Resources

- [ControlNet paper](https://arxiv.org/abs/2302.05543) — for the original idea
- [AnimateAnyone paper](https://arxiv.org/abs/2311.17117) — character-consistent animation
- [MotionCtrl paper](https://arxiv.org/abs/2312.03641)
- [EMO paper](https://arxiv.org/abs/2402.17485) — audio-driven talking heads
- [TokenFlow paper](https://arxiv.org/abs/2307.10373) — consistent video editing
- [Boximator paper](https://arxiv.org/abs/2402.01566)

---

## Phase 8: Long-Form and Consistent Video

The hardest open problem in video generation. Today's best models produce 5–10 seconds of beautiful video and then fall apart. Closing the gap to minute-long, story-coherent generation is the active frontier.

### Concepts to Learn

- **Why long video is hard**:
  - Compute scales at least linearly with length, usually worse
  - Drift: small errors compound; characters morph, scenes contradict themselves
  - Memory: ~30s of latent tokens is already in the 100k–1M range — context window pain
  - No long paired text-video data at scale
- **Sliding-window approaches** — generate overlapping clips, blend in latent or pixel space (FreeNoise, Gen-L-Video)
- **Hierarchical generation**:
  - Generate keyframes first, then fill in between
  - Storyboard / shot decomposition (think a director's storyboard, not raw video)
- **Autoregressive video models** — predict the next chunk of frames conditioned on the previous chunk; long but expensive
- **Anchor frames / scene tokens** — explicit memory of "this character looks like X"
- **Streaming generation** — emit frames as you generate them (StreamingT2V, MovieDreamer)
- **Multi-shot / multi-scene** — VideoTetris, DreamFactory, MovieDreamer; combine LLM-planned shot lists with per-shot generation

### Two Architectural Approaches to Length

```
A. Sliding window with overlap (post-hoc):
   [clip 1: frames 0-15]
       [clip 2: frames 8-23]    ← 8 frames of overlap, blended
              [clip 3: frames 16-31]
                     ...

   + Cheap, works with any existing T2V model
   - Long-range coherence is whatever the overlap can carry forward

B. Hierarchical (designed-in):
   text → LLM "director" → shot list (S1, S2, S3, ...)
                ↓ each shot, per-shot:
        keyframes → fill-in T2V model → 5-sec clip
                ↓ stitch shots
        + consistency model to harmonize identity across shots

   + Can in principle produce minutes of coherent story
   - Three or four separate models; complex to train and orchestrate
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Sliding-window T2V | Take an open T2V model; generate 30 seconds by overlapping 5-sec clips; blend in latent space | ⭐⭐⭐⭐ |
| Keyframe interpolation | Generate 4 keyframes 5 sec apart, then use an I2V or interpolation model to fill in | ⭐⭐⭐⭐ |
| Character consistency | Use a reference-image encoder (IP-Adapter / character LoRA) across multiple shots; measure drift | ⭐⭐⭐⭐⭐ |
| LLM shot planner | Use a small LLM to expand "a knight rescues a princess" into a JSON shot list; generate each shot; evaluate coherence | ⭐⭐⭐⭐⭐ |
| Streaming T2V | Implement chunk-by-chunk generation with a cached KV state across chunks; measure latency vs quality | ⭐⭐⭐⭐⭐ |

### Key Insight

Long-form video generation has the same shape as the long-context problem in LLMs three years ago — exciting demos, brittle outputs, no clear winning architecture, and a half-dozen credible bets. Sliding window, hierarchical planning, and autoregressive generation are not converging the way DiT converged for short video. Expect this to be the dominant frontier topic through 2026–2027.

### Resources

- [FreeNoise paper](https://arxiv.org/abs/2310.15169)
- [StreamingT2V paper](https://arxiv.org/abs/2403.14773)
- [VideoTetris paper](https://arxiv.org/abs/2406.04277)
- [MovieDreamer paper](https://arxiv.org/abs/2407.16655)
- [Long Video Diffusion (NVIDIA)](https://arxiv.org/abs/2407.13759)

---

## Phase 9: World Models and Interactive Video

Where video generation stops being "I make pretty clips" and becomes "I simulate the world."

### Concepts to Learn

- **What a world model is** — a generative model that, given a state and an action, predicts the next state. A video model conditioned on actions is a world model
- **Dreamer-line of work** — Hafner et al.'s DreamerV1/V2/V3, learning world models from pixels for RL
- **Genie, Genie 2 (DeepMind)** — playable, action-conditioned video models trained on web video
- **GameNGen (Google)** — a real-time playable Doom simulation, entirely neural
- **GAIA-1 (Wayve)** — driving world models
- **OASIS / Decart** — open neural Minecraft
- **Latent action models** — inferring actions from unlabeled video (so you can train world models without paired actions)
- **Real-time constraints** — < 50 ms/frame for interactivity. Forces distillation, caching, or smaller models
- **Connection to physical RL** — world models are policy-rollouts-as-video; the same model can serve as a simulator for an RL agent
- **Connection to multimodal** — a fully general world model is multimodal: text in, video out, with audio, actions, and physics

### The World Model Loop

```
                              ┌─────────────────────────┐
                              │                         │
       (state s_t)            │   World model           │
       (action a_t)──────────►│   p(s_{t+1} | s_t, a_t)│──────► (frame s_{t+1})
                              │                         │
                              └─────────────────────────┘
                                          ▲
                                          │
                              (s_{t+1} fed back as next s_t)

Run this in a loop, with actions from a human (interactive game),
an RL policy (sim-for-RL), or a planner (model-based control).

A world model is a video generator that also takes actions —
or equivalently, a video generator IS a world model when "action"
is the empty string.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Action-conditioned video | Take a small video diffusion model; add a discrete-action input (e.g., 4 game actions); train on a simple game's recorded play | ⭐⭐⭐⭐⭐ |
| GameNGen reproduction (mini) | Train an action-conditioned model on a simpler game (Atari, GridWorld) and play it interactively | ⭐⭐⭐⭐⭐ |
| Latent action inference | Train a model to infer the latent action between two adjacent frames in unlabeled video (Genie-style) | ⭐⭐⭐⭐⭐ |
| World model for RL | Use a learned world model to roll out trajectories; train a policy in the dream (DreamerV3-light) | ⭐⭐⭐⭐⭐ |
| Real-time latency hunt | Distill a 30-fps diffusion video model into a 4-step (or 1-step) consistency model; measure ms/frame | ⭐⭐⭐⭐ |

### Key Insight

World models are the convergence point of three lines of research that are usually taught separately: video generation, model-based RL, and simulation. Each of those communities approaches the same object from a different angle — generation people care about visual fidelity, RL people care about action conditioning and rollouts, simulation people care about physical realism. The 2025–2026 frontier is increasingly the same model used in all three roles. If you've completed the [Reinforcement Learning Guide](../reinforcement-learning/) and now this guide, you're well-positioned to work at that intersection.

### Resources

- [Genie 2 (DeepMind)](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) — start here
- [GameNGen paper (Google, 2024)](https://arxiv.org/abs/2408.14837) — playable Doom
- [DreamerV3 paper](https://arxiv.org/abs/2301.04104)
- [GAIA-1 paper](https://arxiv.org/abs/2309.17080)
- [OASIS (Decart) blog](https://www.decart.ai/articles/oasis-interactive-ai-video-game-model)
- [Latent Action Pretraining (Bruce et al.)](https://arxiv.org/abs/2402.15391)

---

## Phase 10: Training at Scale, Evaluation, and Frontier Topics

This last phase is the operational reality of video generation: data, compute, eval, and what's still open.

### Training at Scale

- **Data sources**:
  - Public: HD-VILA, WebVid (deprecated), Panda-70M, OpenVid-1M, Koala-36M
  - Proprietary: most strong models train on private licensed video libraries
- **Caption generation** — public video has terrible captions; recaption with a strong VLM (Qwen2-VL, LLaVA, GPT-4o) before training. This is the single highest-leverage data trick
- **Aspect-ratio bucketing** — train on multiple aspect ratios together for variable-resolution inference
- **Clip extraction** — scene detection + filtering (motion score, aesthetic score, OCR-text score)
- **Curriculum** — start at low resolution and short duration, scale up gradually
- **Compute**: a frontier text-to-video model is on the order of 10²⁵–10²⁶ FLOPs of training; an open replication is 10²³–10²⁴

### Evaluation

The evaluation problem in video generation is *worse* than in image generation, which is already bad.

- **Automatic metrics**:
  - **FVD (Fréchet Video Distance)** — the standard, but criticized for poor correlation with human judgment
  - **CLIPScore-Video, VideoCLIP** — text-video alignment
  - **VBench** — comprehensive benchmark suite; the closest thing to a standard
  - **EvalCrafter** — open evaluation harness
- **Human evaluation** — still the gold standard; pairwise comparisons, win rates
- **Physical correctness** — does water behave like water? Do objects persist when occluded? Largely unmeasured
- **Sora's own evaluation criteria** mention things like "object permanence" and "world consistency" — these still don't have clean benchmarks

### Frontier Topics

- **Real-time video generation** — distillation to 1–4 steps, consistency models, autoregressive caching
- **Audio + video joint generation** — Movie Gen, Veo 2, Lumiere with audio; native AV models instead of post-hoc dubbing
- **Multi-character, multi-scene narratives** — see Phase 8
- **Physical realism** — making fluid behave like fluid, deformable objects deform correctly
- **3D-consistent video** — output that's consistent under camera change (videos that can be re-rendered from a new viewpoint); bridges to NeRF / 3D Gaussian Splatting
- **Editable / re-renderable output** — output something more structured than pixels (e.g., a 3D scene + camera path)
- **Safety**: deepfake detection, watermarking (e.g., SynthID), content moderation, consent
- **Interactive / playable** — see Phase 9
- **Long-context multimodal video** — feeding hours of video to a VLM for understanding; the inverse direction, but adjacent

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Run VBench end to end | Evaluate an open T2V model on the full VBench suite; reproduce a leaderboard number | ⭐⭐⭐ |
| Recaption a dataset | Take 100k clips with bad captions, recaption with a strong VLM, train a small model on each — compare quality | ⭐⭐⭐⭐ |
| Aspect-ratio bucketing | Implement bucketed batching for variable aspect ratios; observe quality improvement on portrait/wide test sets | ⭐⭐⭐ |
| Consistency-model distillation | Distill a 50-step video diffusion model into a 4-step student; measure speed and quality loss | ⭐⭐⭐⭐⭐ |
| Watermarking | Add invisible watermarking to your model's outputs; verify with a detector | ⭐⭐⭐⭐ |
| Physical-plausibility probe | Build 50 trick prompts (water flowing uphill, dropped objects floating); evaluate open models | ⭐⭐⭐ |

### Key Insight

Video generation in 2026 is where text generation was around 2021 — extraordinary demos, frustrating gap to product, two or three competing architectural bets, and absolutely no consensus on evaluation. The compute frontier is rapidly closing in on the data frontier: training a Sora-class model is no longer compute-impossible for many organizations, but obtaining the licensed long-form video to train it on is now the harder problem. If you're entering the field, the highest-leverage skills are not architecture (it's converging on DiT) — they're data engineering, evaluation, and control surfaces.

### Resources

- [VBench paper and leaderboard](https://github.com/Vchitect/VBench)
- [EvalCrafter paper](https://arxiv.org/abs/2310.11440)
- [Movie Gen technical report](https://ai.meta.com/research/movie-gen/) — the most detailed open description of a frontier-scale system
- [Panda-70M dataset](https://snap-research.github.io/Panda-70M/)
- [SynthID for video](https://deepmind.google/technologies/synthid/)

---

## Suggested Timeline

| Phase | Duration | Outcome |
|-------|----------|---------|
| 0. Prerequisites | 0–2 weeks | Image diffusion + multimodal foundations solid |
| 1. Foundations | 1 week | Comfortable loading and decoding video data |
| 2. Classical | 1 week | Familiar with the pre-diffusion approaches; skim only |
| 3. I2V | 1–2 weeks | Built or fine-tuned an image-to-video model |
| 4. Video diffusion | 3 weeks | Inflated a 2D U-Net to 3D and trained on small video |
| 5. Latent + VAE | 2–3 weeks | Trained a 3D VAE; diffusion runs in its latent space |
| 6. DiT | 3–4 weeks | Implemented or ran a DiT-based video model; understand flow matching |
| 7. Conditioning | 2 weeks | Added at least two control signals (camera, depth, character) |
| 8. Long-form | 2–3 weeks | Sliding window or hierarchical pipeline working end to end |
| 9. World models | 2–3 weeks | Trained an action-conditioned model; can roll out interactively |
| 10. Scale + eval | Ongoing | Real benchmark evaluation; data pipeline understood |

**Total to "comfortable practitioner":** ~4–5 months of focused study. Frontier-research-comfortable: closer to a year.

---

## Key Advice

1. **Don't try pixel-space.** Past 64×64 it's wasted compute. Latent space is non-negotiable.
2. **Inflate first, train from scratch later.** Your first video model should reuse pretrained image weights. Going scratch is a frontier-lab activity.
3. **Joint image-video training.** Co-training preserves still-image quality and dramatically helps data efficiency.
4. **Recaption your data.** Web alt-text and YouTube descriptions are terrible. A strong VLM recaptioning your training video is the highest-leverage single change you can make.
5. **The VAE matters as much as the diffusion model.** Bad reconstructions cap your output quality. Spend serious effort here.
6. **Profile decoding.** Most video-training pipelines are bottlenecked on video decoding, not on the GPU. Use `decord`, prefer keyframe-aligned sampling, cache when possible.
7. **`bf16` everywhere on Ampere+.** Same as elsewhere; `float16` GradScalers are unnecessary friction.
8. **Aspect ratios matter.** Train on multiple bucket ratios; resist the urge to center-crop everything to square.
9. **Evaluate with a suite.** Don't trust a single FVD number. Use VBench plus human eval, and report failures honestly.
10. **Watch the open-source frontier.** OpenSora, CogVideoX, HunyuanVideo, Mochi, Wan — these move every few months. The state of "what an individual researcher can run" changes faster here than anywhere else in ML.

---

## Common Pitfalls to Avoid

- ❌ Trying to train pixel-space diffusion at meaningful resolution
- ❌ Using a 2D VAE per frame and being surprised by temporal flicker
- ❌ Ignoring the VAE and treating it as a fixed black box
- ❌ Training only on video and watching still-image quality collapse
- ❌ Loading video with PIL frame by frame instead of `decord`
- ❌ Storing decoded frames as fp32 on disk
- ❌ Using CLIP-L for text conditioning when prompts are >77 tokens (use T5)
- ❌ Reporting only FVD with no human eval
- ❌ Trying to generate >10 seconds without a longform strategy
- ❌ Forgetting to validate frame-by-frame consistency, not just per-frame quality

---

## Additional Resources

### Books and Long-Form Reading
- [Lilian Weng — What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [The Annotated Diffusion Model (Hugging Face)](https://huggingface.co/blog/annotated-diffusion)
- [Sander Dieleman's blog](https://sander.ai/) — best long-form thinking on diffusion

### Key Papers, Chronologically
| Year | Paper | Contribution |
|------|-------|-------------|
| 2022 | [Video Diffusion Models](https://arxiv.org/abs/2204.03458) | First principled paper |
| 2022 | [Make-A-Video](https://arxiv.org/abs/2209.14792) | Text-conditioned video, image-pretrain trick |
| 2022 | [Imagen Video](https://arxiv.org/abs/2210.02303) | Cascaded high-res video |
| 2023 | [Align Your Latents](https://arxiv.org/abs/2304.08818) | Latent video diffusion |
| 2023 | [Stable Video Diffusion](https://arxiv.org/abs/2311.15127) | Open I2V baseline |
| 2023 | [AnimateDiff](https://arxiv.org/abs/2307.04725) | Motion module, community SD |
| 2023 | [MagViT-v2](https://arxiv.org/abs/2310.05737) | Best discrete video tokenizer |
| 2024 | [Sora technical report](https://openai.com/index/video-generation-models-as-world-simulators/) | DiT + variable patches |
| 2024 | [GameNGen](https://arxiv.org/abs/2408.14837) | Real-time neural Doom |
| 2024 | [CogVideoX](https://arxiv.org/abs/2408.06072) | Strong open DiT + VAE |
| 2024 | [Movie Gen](https://ai.meta.com/research/movie-gen/) | Frontier-scale recipe, open description |
| 2024 | [HunyuanVideo](https://arxiv.org/abs/2412.03603) | Large open release |
| 2024 | [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) | Foundation world model |

### Tools You Should Know
- **`decord`** — fast video loading
- **`diffusers` (Hugging Face)** — for inference and quick prototyping
- **`OpenSora` / `CogVideoX` / `HunyuanVideo`** — open training stacks
- **`VBench`** — evaluation harness
- **`comfyui`** — for rapid pipeline prototyping with open models
- **`ffmpeg`** — you will need it

### Communities
- [Hugging Face forums and Discord](https://discuss.huggingface.co/)
- [r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/) — practical, open-source-focused
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- Twitter/X — video gen moves on Twitter faster than anywhere

---

## Quick Start Checklist

- [ ] Can load a video clip with `decord` and explain frame sampling vs uniform-in-time
- [ ] Can explain why latent space is mandatory for video generation
- [ ] Have run inference on Stable Video Diffusion and AnimateDiff
- [ ] Have inflated a 2D U-Net to a 3D (or (2+1)D) model and trained it on small video
- [ ] Have trained or used a 3D VAE; understand causal video VAEs
- [ ] Have read the Sora technical report end to end
- [ ] Have implemented or run a DiT-based video model
- [ ] Understand flow matching as well as DDPM
- [ ] Can add a control signal (camera, depth, pose) to a video model
- [ ] Have generated >10 sec of video with a longform strategy
- [ ] Have evaluated a model on VBench (or a substantial subset)
- [ ] Have at least skimmed an action-conditioned world model paper (Genie 2 or GameNGen)

---

## License

MIT License. See the [LICENSE](https://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
