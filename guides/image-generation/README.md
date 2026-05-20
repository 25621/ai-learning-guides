# Image Generation: From Beginner to Advanced

A comprehensive guide to understanding and building image generation systems — from the underlying problem of modeling a distribution over pixels, through autoencoders, GANs, and diffusion, to the modern frontier of latent diffusion transformers, flow matching, and controllable generation.

> **Image generation is density estimation in disguise.** You are trying to learn `p(image)` — or more usefully, `p(image | text)` — over a space with millions of dimensions, where almost every point is noise and the manifold of "real-looking pictures" is vanishingly thin. The history of the field is the history of finding tractable surrogates for that intractable objective: adversarial games, variational bounds, denoising scores. Diffusion won; understanding *why* it won is the point of this guide.

---

## Table of Contents

1. [Phase 0: Prerequisites](#phase-0-prerequisites)
2. [Phase 1: Foundations — Images, Likelihoods, and the Manifold Problem](#phase-1-foundations--images-likelihoods-and-the-manifold-problem)
3. [Phase 2: Autoencoders and VAEs](#phase-2-autoencoders-and-vaes)
4. [Phase 3: Discrete Latents — VQ-VAE, VQ-GAN, and Modern Tokenizers](#phase-3-discrete-latents--vq-vae-vq-gan-and-modern-tokenizers)
5. [Phase 4: Generative Adversarial Networks](#phase-4-generative-adversarial-networks)
6. [Phase 5: Diffusion Models — Foundations (DDPM)](#phase-5-diffusion-models--foundations-ddpm)
7. [Phase 6: Score-Based, EDM, and Modern Diffusion Theory](#phase-6-score-based-edm-and-modern-diffusion-theory)
8. [Phase 7: Latent Diffusion and Stable Diffusion](#phase-7-latent-diffusion-and-stable-diffusion)
9. [Phase 8: Diffusion Transformers and Flow Matching](#phase-8-diffusion-transformers-and-flow-matching)
10. [Phase 9: Conditioning, Control, and Personalization](#phase-9-conditioning-control-and-personalization)
11. [Phase 10: Training at Scale, Distillation, Evaluation, and Frontier Topics](#phase-10-training-at-scale-distillation-evaluation-and-frontier-topics)
12. [Suggested Timeline](#suggested-timeline)
13. [Key Advice](#key-advice)
14. [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
15. [Additional Resources](#additional-resources)
16. [Glossary](#glossary)

---

## Phase 0: Prerequisites

Image generation borrows heavily from probability theory, optimization, and convolutional/transformer architectures. None of it is exotic, but if more than a couple of these are fuzzy, fix that first.

### Concepts to Know

- **PyTorch fluency**: `nn.Module`, autograd, mixed precision, basic training loops, profiling
- **Convolutional networks**: stride/padding/dilation, ResNet-style residual blocks, batch and group normalization
- **Transformers**: self-attention, cross-attention, positional embeddings, layer norm
- **Probability**: joint vs conditional distributions, KL divergence, expectation, Jensen's inequality
- **Calculus**: chain rule, gradients of a scalar w.r.t. a vector, the reparameterization trick (you'll see it again)
- **Linear algebra**: matrix-vector products, eigendecomposition (helpful for understanding noise schedules)
- **Information theory** (light touch): entropy, cross-entropy, the bits-per-dimension metric

### The One Equation Everything Comes Back To

```
The generative modeling problem:

    Given samples x ~ p_data(x), learn a model p_θ(x) such that
    sampling from p_θ produces new x's that look like they came from p_data.

The four approaches we'll study, in one line each:

    VAEs:     maximize a lower bound on log p_θ(x) via an encoder q(z|x).
    GANs:     don't model p_θ(x) at all; just learn to fool a classifier.
    Diffusion: model p_θ(x) as the reverse of a fixed noising process.
    Flow:     model p_θ(x) as the endpoint of a learned ODE/velocity field.

Diffusion and flow are the same thing under different parameterizations.
GANs and VAEs are the historical record. You should understand all four.
```

### Resources

- [PyTorch Deep Dive guide](../pytorch-deep-dive/) — strongly recommended prerequisite
- [Lilian Weng — What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — the canonical primer; reread it once per phase
- [Goodfellow, Bengio, Courville — Deep Learning, Ch. 20](https://www.deeplearningbook.org/contents/generative_models.html) — slightly dated but the foundational framing still holds
- [Sander Dieleman's blog](https://sander.ai/) — the best long-form writing on diffusion, by one of the people who built it

---

## Phase 1: Foundations — Images, Likelihoods, and the Manifold Problem

Before models, understand the problem. The intuition you build here will tell you why every later trick exists.

### Concepts to Learn

- **Images as tensors**: `(B, C, H, W)`; range `[0, 255]` (uint8) vs `[0, 1]` vs `[-1, 1]` (the model-friendly default)
- **The dimensionality problem**: a 256×256 RGB image lives in ~200,000-dimensional space. Almost every point in that space is noise
- **The manifold hypothesis**: real images lie on a vanishingly thin manifold inside pixel space; generation is learning that manifold
- **Likelihood-based vs implicit models**: do you assign a number `p(x)` to each image, or do you only learn to *sample*?
- **Mode collapse vs mode covering**: what each loss function rewards. KL(q‖p) covers modes; KL(p‖q) seeks modes. This single asymmetry explains a lot
- **Bits per dimension (bpd)** — the standard likelihood metric on images
- **FID, IS, KID, CLIPScore** — the standard quality metrics. All of them are flawed; you will use them anyway
- **Why direct maximum likelihood is hard**: tractable density estimators (normalizing flows, autoregressive pixel models) trade quality for likelihood; the best-looking models don't even have a tractable likelihood
- **Autoregressive pixel models** (PixelRNN, PixelCNN, PixelCNN++): the path-not-taken; conceptually clean, painfully slow at sampling
- **Normalizing flows** (Real NVP, Glow): tractable exact likelihood, but architecturally constrained; mostly historical now

### A Map of Generative Model Families

```
                           GENERATIVE MODELS
                                 │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
    LIKELIHOOD-BASED        IMPLICIT             SCORE / DIFFUSION
    (tractable density)     (sampling only)      (modern winner)
        │                       │                       │
   ┌────┴────┐                  GANs              ┌─────┴─────┐
   │         │                  │                 │           │
 PixelCNN/  Normalizing       (no density;        DDPM       Flow matching
 PixelRNN   flows              fool a critic)    Score SDE   Rectified flow
 (autoreg)  (Real NVP, Glow)
   │         │
   VAEs ────┘
   (variational bound — between flows and GANs)

Diffusion/flow models are technically likelihood-trainable, but in
practice nobody cares about the likelihood — they care about samples.
That's why the metric of choice is FID, not bits-per-dim.
```

### The Cost of an Image

```
Resolution × channels → raw pixel count

  32×32×3       →  3,072 dims     (CIFAR-10: where most papers prototype)
  64×64×3       →  12,288 dims    (CelebA, ImageNet-64)
  256×256×3     →  196,608 dims   (the "standard" research resolution)
  512×512×3     →  786,432 dims   (Stable Diffusion-era default)
  1024×1024×3   →  3,145,728 dims (SDXL, modern T2I models)

Doubling resolution = 4× pixels = 4× memory = ~4× compute per forward pass.
This is the same scaling pain that drives latent diffusion in Phase 7.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Manifold visualizer | Draw 1,000 CIFAR-10 images and 1,000 uniform-random images; PCA both into 2D; observe the manifold | ⭐⭐ |
| Bits-per-dim baseline | Compute the bits-per-dim of a uniform distribution over `[0, 255]^d` on a small image dataset; this is the "no model" floor | ⭐⭐ |
| Tiny PixelCNN | Implement a small autoregressive pixel model on MNIST; sample row by row; observe the speed (slow!) and quality (decent) | ⭐⭐⭐ |
| FID from scratch | Implement Fréchet Inception Distance: extract Inception features, compute means/covariances, plug into the closed-form formula | ⭐⭐⭐ |
| Real NVP toy | Implement a small normalizing flow on 2D toy data (moons, swiss roll); visualize learned density vs samples | ⭐⭐⭐ |

### Sample Code: Loading and Normalizing Images for a Generative Model

```python
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Standard preprocessing for generative models: scale to [-1, 1].
# (Discriminative models often use ImageNet mean/std normalization;
# generative models do not, because the model has to output pixels
# that round-trip through the same transform at sample time.)
transform = T.Compose([
    T.ToTensor(),                                  # (C, H, W) in [0, 1]
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),              # → [-1, 1]
])

train = CIFAR10(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True)

# To go back to displayable pixels:
def to_uint8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
```

### Key Insight

Every generative model is a fight against the manifold. Pixel space is enormous; the manifold of believable images is a thread through it. Likelihood-based models (PixelCNN, flows) get punished for assigning *any* probability to off-manifold points, even if those points are visually irrelevant — which is why they spend capacity on the wrong things and produce blurry samples despite high likelihood. Implicit models (GANs) don't have this problem because they never have to assign a density anywhere; they just have to put samples on the manifold. Diffusion's elegance is that it gets the best of both: tractable training (a regression loss on noise) and samples that land on the manifold (because the noising process defines the manifold by construction).

### Resources

- [Lilian Weng — Flow-based Deep Generative Models](https://lilianweng.github.io/posts/2018-10-13-flow-models/)
- [PixelCNN++ paper (Salimans et al., 2017)](https://arxiv.org/abs/1701.05517)
- [Are GANs Created Equal? (Lucic et al., 2017)](https://arxiv.org/abs/1711.10337) — the original FID skepticism paper
- [Heusel et al. — original FID paper](https://arxiv.org/abs/1706.08500)
- [Real NVP paper](https://arxiv.org/abs/1605.08803)

---

## Phase 2: Autoencoders and VAEs

Autoencoders are the workhorse of generative vision — not always as the *generator*, but always somewhere in the pipeline. By the end of this phase you should be able to explain why every modern image model has a VAE buried inside it.

### Concepts to Learn

- **Vanilla autoencoders**: encoder → bottleneck → decoder; reconstruction loss; latent space is *learned* but unstructured
- **The collapse problem**: a high-capacity AE just memorizes; the bottleneck dimension is the regularizer
- **Variational autoencoders (VAE)**: encoder outputs `q(z|x) = N(μ, σ²)`, decoder samples; trained on the ELBO
- **The ELBO**: reconstruction term + KL term to a prior `N(0, I)`. Why the KL term is what makes it a *generative* model
- **The reparameterization trick**: `z = μ + σ·ε`, `ε ~ N(0, I)` — required for gradients to flow through the sample
- **Posterior collapse**: when the decoder is too strong, the KL term pushes `q(z|x)` to the prior, and the model ignores `z`. The β-VAE family explicitly tunes this trade-off
- **Why VAE samples are blurry**: pixel-MSE reconstruction + Gaussian decoder = blur; the model averages over plausible reconstructions
- **β-VAE, NVAE, VQ-VAE-2** as a family of fixes for the blur and the structure of the latent
- **Hierarchical VAEs**: multiple latent levels at different resolutions; how NVAE and Very Deep VAE squeeze quality out of the family
- **The VAE-as-compressor view**: even if a VAE makes bad samples, it makes a great latent space for *another* model (diffusion!) to generate in. This is the role the VAE plays in modern systems

### The VAE Picture

```
          ┌─────────┐                      ┌─────────┐
   x ───► │ encoder │ ──► μ, log σ²  ──┐   │ decoder │ ──► x̂
          └─────────┘                  │   └─────────┘
                                       │       ▲
                                       ▼       │
                              z = μ + σ · ε ───┘
                                  ε ~ N(0, I)            ← reparam trick

Loss = ||x - x̂||²                                       ← reconstruction
     + β · KL(q(z|x) ‖ p(z))    where p(z) = N(0, I)    ← regularization

At sample time:
    z ~ N(0, I)
    x_new = decoder(z)
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Tiny AE on MNIST | 28×28 → 32-dim latent → 28×28; observe that linear interpolation in latent space gives reasonable mid-digits | ⭐⭐ |
| Vanilla VAE | Same dataset, full ELBO; sample from `N(0, I)` and decode; compare blur vs the AE | ⭐⭐⭐ |
| β-VAE study | Sweep β from 0 to 10; observe the trade-off between reconstruction and KL; identify posterior collapse | ⭐⭐⭐ |
| Conditional VAE | Add class conditioning; verify you can sample a specific MNIST digit | ⭐⭐⭐ |
| Hierarchical VAE | Two-level latent (one 8×8, one 4×4); see whether quality improves on CIFAR-10 | ⭐⭐⭐⭐ |
| Latent traversal | Take a trained VAE on CelebA; walk along single latent dimensions; identify ones that control hair, smile, lighting | ⭐⭐⭐ |

### Sample Code: A VAE ELBO Step

```python
import torch
import torch.nn.functional as F

def vae_step(encoder, decoder, x, beta=1.0):
    # Encoder outputs the parameters of q(z|x)
    mu, log_var = encoder(x).chunk(2, dim=1)            # both (B, D)

    # Reparameterized sample
    std = (0.5 * log_var).exp()
    eps = torch.randn_like(std)
    z = mu + std * eps

    # Decode and compute reconstruction
    x_hat = decoder(z)
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)

    # KL(q(z|x) || N(0, I)) — closed form for diagonal Gaussians
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()

    loss = recon + beta * kl
    return loss, {"recon": recon.item(), "kl": kl.item()}
```

### Key Insight

The VAE was the dominant generative model of 2014–2016, lost the spotlight to GANs in 2017–2019, and lost it again to diffusion in 2020. But the VAE never went away — it just stopped being the generator. Today's strongest image (and video, and audio) generators all train diffusion in the latent space of a VAE. The VAE compresses 256×256×3 pixels into a 32×32×4 latent that's ~48× smaller, and the diffusion model gets to work on the small thing. The story of generative modeling in the last decade is partly the story of better compressors enabling better generators.

### Resources

- [Kingma & Welling — Auto-Encoding Variational Bayes (2013)](https://arxiv.org/abs/1312.6114) — the original VAE
- [β-VAE paper](https://openreview.net/forum?id=Sy2fzU9gl)
- [NVAE: A Deep Hierarchical VAE (Vahdat & Kautz, 2020)](https://arxiv.org/abs/2007.03898)
- [Very Deep VAEs (Child, 2020)](https://arxiv.org/abs/2011.10650)
- [Doersch — Tutorial on VAEs](https://arxiv.org/abs/1606.05908) — the clearest pedagogical write-up

---

## Phase 3: Discrete Latents — VQ-VAE, VQ-GAN, and Modern Tokenizers

Continuous latents give you diffusion. Discrete latents give you transformers, language-model-style generation, and the "image is just a sequence of tokens" framing. Both branches matter.

### Concepts to Learn

- **VQ-VAE**: replace continuous `z` with a finite codebook lookup; gradients via the straight-through estimator
- **The three losses**: reconstruction, codebook loss, commitment loss; what each term is responsible for
- **Codebook collapse**: when only a few entries get used. Mitigations: EMA codebook updates, dead-code re-init, k-means warmup
- **VQ-VAE-2**: hierarchical discrete latents; the first VQ system that scaled to high-resolution natural images
- **VQ-GAN / Taming Transformers**: replace pixel-MSE with LPIPS + a patch discriminator. This is the trick that made VQ models actually look good — and the recipe Stable Diffusion's VAE descends from
- **RQ-VAE (Residual Quantization)**: multiple codebook lookups per spatial position; better information packing
- **FSQ (Finite Scalar Quantization)**: codebook-free quantization; bins each latent coordinate independently. Surprisingly competitive with much simpler training
- **LFQ (Lookup-Free Quantization)** and **MagViT-v2**: binary quantization per dim; the current open-state-of-the-art for video tokenization, and increasingly for images too
- **Autoregressive image generation from discrete tokens**:
  - Raster-order transformers (the original DALL-E)
  - Parti, Muse — masked-token / parallel-decode variants
  - Modern token-based image gen (LlamaGen, Show-o, Emu3) — "GPT for images"
- **The continuous-vs-discrete debate** for image generation in 2026: diffusion still leads on quality, but discrete-token approaches are catching up and integrate naturally with LLMs (any-to-any models)

### The VQ-VAE Picture

```
          ┌─────────┐
   x ───► │ encoder │ ──► z_e (continuous)
          └─────────┘
                        │
                        ▼
              find nearest codebook entry e_k:
                  z_q = e_k                              ← discrete latent
                        │
                        ▼
                  ┌─────────┐
                  │ decoder │ ──► x̂
                  └─────────┘

Loss = ||x - x̂||²                                       ← reconstruction
     + ||sg[z_e] - e_k||²                                ← codebook update
     + β · ||z_e - sg[e_k]||²                            ← commitment

(sg = stop-gradient; gradients through z_q flow as if it were z_e.)
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| VQ-VAE on CIFAR-10 | 256-entry codebook, 8×8 latent grid per image; visualize codebook usage and reconstructions | ⭐⭐⭐⭐ |
| Codebook collapse hunt | Train VQ with a too-large codebook; identify and fix collapse via EMA updates and dead-code re-init | ⭐⭐⭐ |
| VQ-GAN | Add LPIPS + patch discriminator to your VQ-VAE; observe reconstructions go from blurry to sharp | ⭐⭐⭐⭐ |
| FSQ tokenizer | Implement finite-scalar quantization (no learned codebook); compare to VQ on reconstruction FID | ⭐⭐⭐⭐ |
| Tiny image transformer | Train a small autoregressive transformer over the VQ-GAN tokens of CIFAR-10; sample row by row | ⭐⭐⭐⭐⭐ |
| Masked-token model | Implement a MaskGIT-style parallel decoder over the same tokens; compare sampling speed and quality | ⭐⭐⭐⭐⭐ |

### Sample Code: A VQ Codebook Lookup with the Straight-Through Estimator

```python
import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment=0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings,
                                            1.0 / num_embeddings)
        self.beta = commitment

    def forward(self, z_e):
        # z_e: (B, D, H, W). Flatten to (B*H*W, D) for nearest-neighbor lookup.
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)

        # Squared distance to each codebook entry
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.codebook.weight.t()
             + self.codebook.weight.pow(2).sum(1))
        idx = d.argmin(dim=1)
        z_q = self.codebook(idx).view(B, H, W, D).permute(0, 3, 1, 2)

        # Codebook + commitment losses
        codebook_loss   = (z_q.detach() - z_e).pow(2).mean()
        commitment_loss = (z_q - z_e.detach()).pow(2).mean()
        loss = commitment_loss + self.beta * codebook_loss

        # Straight-through: gradients flow as if z_q == z_e
        z_q = z_e + (z_q - z_e).detach()
        return z_q, loss, idx
```

### Key Insight

Once you tokenize an image — turn it into a discrete sequence with a fixed vocabulary — it becomes "just another language" for a transformer. This is the bet behind Parti, Emu3, Chameleon, and the entire "any-to-any" line of research: if you can tokenize every modality (image, audio, video, action), one transformer can model the joint distribution with one next-token-prediction loss. The current state of play is that this works, but discrete-token image gen still trails diffusion on raw quality at the same scale. The frontier is closing fast.

### Resources

- [VQ-VAE paper (van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937)
- [VQ-VAE-2 paper (Razavi et al., 2019)](https://arxiv.org/abs/1906.00446)
- [VQ-GAN / Taming Transformers (Esser et al., 2020)](https://arxiv.org/abs/2012.09841)
- [MaskGIT paper (Chang et al., 2022)](https://arxiv.org/abs/2202.04200)
- [Parti paper (Yu et al., 2022)](https://arxiv.org/abs/2206.10789)
- [FSQ paper](https://arxiv.org/abs/2309.15505)
- [MagViT-v2 paper](https://arxiv.org/abs/2310.05737)
- [LlamaGen (Sun et al., 2024)](https://arxiv.org/abs/2406.06525) — "GPT for images" recipe

---

## Phase 4: Generative Adversarial Networks

GANs lost the throne to diffusion around 2021, but understanding them is still essential. Many ideas (perceptual losses, style-based generators, truncation tricks) live on inside modern systems.

### Concepts to Learn

- **The adversarial game**: a generator `G` maps noise `z → x`; a discriminator `D` tries to tell real from fake; both train against each other
- **The original GAN objective**: min-max over `log D(x) + log(1 - D(G(z)))`; vanishing gradients early in training
- **Non-saturating loss**: the practical reformulation; what everyone actually uses
- **Mode collapse**: `G` finds one or a few outputs that fool `D` and stops exploring. The defining failure mode
- **DCGAN**: the first GAN architecture that worked reliably; transposed convolutions, batch norm, ReLU/LeakyReLU
- **Wasserstein GAN (WGAN, WGAN-GP)**: a different loss (Earth Mover's Distance) with much better training stability; gradient penalty as the practical regularizer
- **Progressive growing (PGGAN)**: train at low resolution, gradually add layers — predecessor of StyleGAN
- **StyleGAN, StyleGAN2, StyleGAN3**: style-based architecture, adaptive instance normalization, the W/W+ latent space; the "you've seen these on `thispersondoesnotexist.com`" models
- **BigGAN**: the conditional-ImageNet model; truncation trick; class-conditional batch norm
- **Spectral normalization**: bounds the Lipschitz constant of `D`; essential stabilizer
- **R1 regularization** (Mescheder et al.): the simpler stabilizer that StyleGAN2 uses
- **Conditional GANs**: cGAN, AC-GAN, projection discriminator; the path to text-to-image via early systems like AttnGAN and the autoregressive original DALL-E
- **GAN inversion**: given a real image, find the `z` such that `G(z) ≈ image` — both for editing and as a baseline for "controllable" image gen

### The Two-Player Game

```
                     ┌─────────────────────────────────────┐
   z ~ N(0, I) ────► │           Generator G               │ ──► x_fake
                     └─────────────────────────────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
   x_real ──────────────► │      Discriminator D    │ ──► [0, 1]
                          └─────────────────────────┘
                                       ▲
                                       │
                              "real" or "fake"?

Train D to maximize:    E[log D(x_real)] + E[log(1 - D(G(z)))]
Train G to minimize:    E[log(1 - D(G(z)))]                       ← saturating
   or, in practice:     -E[log D(G(z))]                            ← non-saturating

Steps alternate (usually 1:1, sometimes K:1 for D-heavy schedules).
At equilibrium: D(x) ≈ 0.5 everywhere; G's distribution matches the data.
In practice: equilibrium is rarely reached; you stop at the prettiest checkpoint.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Vanilla GAN on MNIST | Implement DCGAN; train; observe mode collapse risk and oscillating losses | ⭐⭐⭐ |
| WGAN-GP | Replace the loss with Wasserstein + gradient penalty; observe training stability improvement | ⭐⭐⭐⭐ |
| Conditional GAN | Add class conditioning via projection discriminator; train on CIFAR-10; verify class control | ⭐⭐⭐ |
| StyleGAN tour | Run inference on a pretrained StyleGAN2 face model; explore W vs W+ latent edits | ⭐⭐ |
| GAN inversion | Given a real image, find the `z` such that `G(z) ≈ image` — via optimization, then via a learned encoder | ⭐⭐⭐⭐ |
| FID head-to-head | Train a DCGAN and a tiny VAE on the same dataset; compare FID, training time, and stability | ⭐⭐⭐ |

### Sample Code: The Non-Saturating GAN Loss

```python
import torch
import torch.nn.functional as F

def discriminator_step(D, G, x_real, z, opt_D):
    opt_D.zero_grad()
    # Real
    d_real = D(x_real)
    loss_real = F.binary_cross_entropy_with_logits(
        d_real, torch.ones_like(d_real))
    # Fake (detached so G doesn't get updates here)
    with torch.no_grad():
        x_fake = G(z)
    d_fake = D(x_fake)
    loss_fake = F.binary_cross_entropy_with_logits(
        d_fake, torch.zeros_like(d_fake))
    loss = loss_real + loss_fake
    loss.backward()
    opt_D.step()
    return loss.item()

def generator_step(D, G, z, opt_G):
    opt_G.zero_grad()
    x_fake = G(z)
    d_fake = D(x_fake)
    # Non-saturating: maximize log D(fake), i.e. label-as-real BCE
    loss = F.binary_cross_entropy_with_logits(
        d_fake, torch.ones_like(d_fake))
    loss.backward()
    opt_G.step()
    return loss.item()
```

### Key Insight

The GAN era taught the field three things that outlasted GANs themselves: (1) perceptual losses (LPIPS, learned features from a classifier) beat pixel MSE for almost every image task; (2) high-quality samples don't require an explicit likelihood — and tractable likelihood often *hurts* sample quality; (3) regularizing the discriminator (spectral norm, gradient penalty, R1) is harder and more important than designing the generator. Diffusion inherited all three lessons. Stable Diffusion's VAE is a VQ-GAN; modern flow models still use LPIPS in their VAEs; and the "discriminator" intuition (don't let any classifier reliably distinguish your samples from real ones) became the implicit reward shaping behavior of CFG and classifier guidance.

### Resources

- [Goodfellow et al. — Generative Adversarial Networks (2014)](https://arxiv.org/abs/1406.2661) — the original
- [Radford et al. — DCGAN (2015)](https://arxiv.org/abs/1511.06434)
- [Arjovsky et al. — Wasserstein GAN (2017)](https://arxiv.org/abs/1701.07875) and [WGAN-GP (Gulrajani et al., 2017)](https://arxiv.org/abs/1704.00028)
- [Karras et al. — Progressive GANs (2017)](https://arxiv.org/abs/1710.10196)
- [StyleGAN paper (2018)](https://arxiv.org/abs/1812.04948) and [StyleGAN2 (2019)](https://arxiv.org/abs/1912.04958)
- [Brock et al. — BigGAN (2018)](https://arxiv.org/abs/1809.11096)
- [Miyato et al. — Spectral Normalization (2018)](https://arxiv.org/abs/1802.05957)
- [Mescheder et al. — R1 regularization (2018)](https://arxiv.org/abs/1801.04406)

---

## Phase 5: Diffusion Models — Foundations (DDPM)

This is the phase to master deeply. Everything in modern image (and video, and audio) generation builds on what's here.

### Concepts to Learn

- **The forward process**: progressively add Gaussian noise to a real image over `T` steps until it's pure noise. *Fixed*, not learned
- **The reverse process**: train a network to denoise one step at a time. This is what's learned
- **The noise schedule**: how much noise per step. Linear vs cosine vs sigmoid schedules
- **The DDPM objective**: predict the noise `ε` added to a noised image `x_t` given `t`. A simple MSE regression
- **The closed-form forward step**: `x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε`. You can jump to any `t` in one step — crucial for efficient training
- **Three equivalent prediction targets**: predict `ε` (DDPM), predict `x_0` (some papers), or predict velocity `v` (progressive distillation). Pick one; the others fall out by algebra
- **DDPM sampling** — stochastic, original; many steps
- **DDIM** — deterministic, much fewer steps; the standard for years
- **The U-Net architecture for diffusion**:
  - Down-blocks with self-attention at lower resolutions
  - Up-blocks with skip connections
  - Time embedding via sinusoidal + MLP, injected via AdaGN / FiLM
  - Cross-attention for conditioning (text via CLIP/T5 embeddings)
- **Class-conditional diffusion**: the ImageNet-scale milestone (Dhariwal & Nichol)
- **Classifier guidance** (pre-CFG): use a separately-trained classifier to gradient-steer samples

### The Forward Process, in One Equation

```
Forward (fixed, no learning):
    q(x_t | x_{t-1}) = N(x_t; √(1 - β_t) · x_{t-1}, β_t · I)

Marginal (closed-form jump to any t):
    q(x_t | x_0) = N(x_t; √(ᾱ_t) · x_0, (1 - ᾱ_t) · I)
    where  ᾱ_t = Π_{s=1..t} (1 - β_s)

So given a real x_0 and a sampled ε ~ N(0, I):
    x_t = √(ᾱ_t) · x_0  +  √(1 - ᾱ_t) · ε              ← cheap, one-shot

Reverse (learned):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_t)

Loss (DDPM, simplified):
    L = E_{x_0, t, ε} ‖ ε - ε_θ(x_t, t) ‖²

i.e., regression on the added noise. That's the entire loss.
```

### The U-Net at a Glance

```
       (B, C, H, W) input                                    output (predicted noise)
              │                                                      ▲
              ▼                                                      │
        ┌─────────────┐                                       ┌──────────────┐
        │ in_conv     │                                       │ out_conv     │
        └─────┬───────┘                                       └──────┬───────┘
              │                                                      │
        ┌─────┴───────┐   ←─── skip ────────────────────────► ┌──────┴───────┐
        │ Down block  │                                       │  Up block    │
        │ (res + attn)│                                       │ (res + attn) │
        └─────┬───────┘                                       └──────┬───────┘
              │                                                      ▲
        ┌─────┴───────┐   ←─── skip ─────────────────────────►┌──────┴───────┐
        │ Down block  │                                       │  Up block    │
        └─────┬───────┘                                       └──────┬───────┘
              │                                                      ▲
              ▼                                                      │
                            ┌────────────────────────┐
                            │   Middle block         │
                            │ (res + attn + cross-attn)
                            └────────────────────────┘

Each block also receives:
    - time embedding t  → MLP → AdaGN / FiLM modulation
    - text embedding c  → cross-attention keys/values
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| DDPM on MNIST | Implement the forward process and a small U-Net; train; sample with the full T-step loop | ⭐⭐⭐ |
| DDPM on CIFAR-10 | Scale up to 32×32 color images; reach FID < 20; this is the "you understand it" milestone | ⭐⭐⭐⭐ |
| Cosine vs linear schedule | Train two otherwise-identical DDPMs; compare FID and visual quality | ⭐⭐⭐ |
| DDIM sampler | Add DDIM sampling to your DDPM; verify 50 steps ≈ 1000-step DDPM quality | ⭐⭐⭐ |
| Class-conditional DDPM | Add label-conditional batch norm or AdaGN; train on CIFAR-10 with labels | ⭐⭐⭐⭐ |
| Classifier guidance | Train a classifier on noisy CIFAR-10 images; use its gradients to steer DDPM samples | ⭐⭐⭐⭐ |

### Sample Code: The DDPM Training Step

```python
import torch
import torch.nn.functional as F

class DDPMTrainer:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.T = T
        # Linear schedule (cosine usually works better; left as exercise)
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)         # ᾱ_t

    def q_sample(self, x0, t, noise):
        """Forward noising in one shot using the closed-form marginal."""
        a_bar = self.alpha_bar[t].view(-1, 1, 1, 1).to(x0.device)
        return a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise

    def loss(self, x0, cond=None):
        B = x0.size(0)
        # Sample a timestep per example
        t = torch.randint(0, self.T, (B,), device=x0.device)
        # Sample noise and form x_t
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps)
        # Predict the noise
        eps_pred = self.model(x_t, t, cond)
        return F.mse_loss(eps_pred, eps)
```

### Key Insight

The thing that makes diffusion *work* — and the thing that took the field years to articulate clearly — is that it converts an intractable density-estimation problem into a *regression* problem with a perfectly clean signal. There is no adversarial game (so no mode collapse), no variational bound (so no posterior collapse), no autoregressive sequence (so it parallelizes), and no normalizing-flow constraint on the architecture (so any U-Net works). The forward process is a free, infinite source of paired training data: noise some image, predict the noise. It's almost embarrassing how simple the loss is. Every "trick" we'll see in the next phase is a refinement of that single core idea.

### Resources

- [Ho et al. — Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239) — the paper that started the diffusion era
- [Nichol & Dhariwal — Improved DDPM (2021)](https://arxiv.org/abs/2102.09672) — cosine schedule, learned variances
- [Dhariwal & Nichol — Diffusion Models Beat GANs (2021)](https://arxiv.org/abs/2105.05233) — the FID-on-ImageNet moment; classifier guidance
- [Song et al. — DDIM (2020)](https://arxiv.org/abs/2010.02502)
- [Lilian Weng — What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

---

## Phase 6: Score-Based, EDM, and Modern Diffusion Theory

DDPM is the right place to start, but the modern field has reformulated it in cleaner mathematical language. Master this phase and you can read any 2022+ diffusion paper without translation overhead.

### Concepts to Learn

- **The score function**: `∇_x log p(x)` — the gradient of log-density with respect to inputs
- **Score matching**: learn the score directly by matching it to a fixed target (denoising score matching, sliced score matching)
- **Score-based generative modeling (Song & Ermon)**: noisy datasets at multiple scales, Langevin dynamics sampling
- **The SDE/ODE view (Song et al., 2021)**: diffusion is the discretization of an SDE; the reverse process can be solved as an ODE (deterministic) or SDE (stochastic)
- **Probability flow ODE**: the deterministic equivalent of the reverse SDE; enables exact likelihood computation
- **Variance Preserving (VP) vs Variance Exploding (VE) SDEs**: the two families; DDPM is VP, score-based papers were originally VE
- **The σ-schedule (EDM, Karras et al. 2022)**: parameterize diffusion by noise standard deviation σ rather than discrete step `t`; cleaner formulation, simpler tuning
- **EDM preconditioning**: scale inputs, outputs, and time so the network sees roughly unit-variance everything regardless of σ. The single biggest "you should have done this from the start" trick of recent years
- **Higher-order ODE solvers** for sampling:
  - **DPM-Solver, DPM-Solver++** — multistep methods designed for diffusion ODEs
  - **Heun's method** (EDM's default) — predictor-corrector at 2nd order
  - **Euler** — the simplest first-order baseline
- **Classifier-free guidance (CFG)**: train conditionally and unconditionally; at inference, push samples away from the unconditional and toward the conditional. The single biggest practical trick in modern T2I
- **Why CFG works**: it sharpens `p(x | c)` toward modes where `p(c | x)` is high — an implicit Bayes-rule manipulation
- **The schedule does not matter as much as you'd think**: a well-tuned EDM model with a few different schedules gets similar FID

### Three Equivalent Pictures of the Same Object

```
DDPM (discrete steps):
    x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε,   t = 0, 1, ..., T

Continuous SDE (Song et al.):
    dx = f(x, t) dt + g(t) dw           ← forward (noising) SDE
    dx = [f(x,t) - g(t)² · s_θ(x,t)] dt + g(t) dw̄   ← reverse (sampling)

σ-space (EDM):
    Sample σ ~ p_σ
    x_σ = x_0 + σ · ε,   ε ~ N(0, I)
    Loss: λ(σ) · || D_θ(x_σ, σ) - x_0 ||²    where λ(σ) makes the loss σ-balanced

All three describe the same family of models. EDM is the cleanest; if
you're starting fresh today, use the σ-parameterization.
```

### Sample Code: Classifier-Free Guidance at Inference

```python
@torch.no_grad()
def sample_with_cfg(model, x_t, t, cond, scale=7.5):
    """One denoising step with CFG."""
    # Run the model twice: once conditional, once unconditional
    eps_cond   = model(x_t, t, cond)
    eps_uncond = model(x_t, t, None)
    # Push away from unconditional, toward conditional
    return eps_uncond + scale * (eps_cond - eps_uncond)
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Score matching from scratch | Implement denoising score matching on 2D toy data (8-Gaussians, swiss roll); verify with Langevin sampling | ⭐⭐⭐ |
| Higher-order sampler | Implement DPM-Solver++ or Heun for an existing DDPM; verify 10–20 steps is enough | ⭐⭐⭐⭐ |
| Classifier-free guidance | Add label-dropout training (10% null condition); implement CFG at inference; sweep the scale 1.0 → 12.0 | ⭐⭐⭐ |
| EDM reparameterization | Re-derive your CIFAR-10 model in the Karras σ-space; verify it trains and samples; observe the cleaner hyperparameter surface | ⭐⭐⭐⭐ |
| Probability flow ODE | Convert your SDE sampler to the deterministic ODE; verify samples and compute exact log-likelihood | ⭐⭐⭐⭐⭐ |
| VP vs VE comparison | Train the same model under both SDE families; compare FID, training stability, and sampler behavior | ⭐⭐⭐⭐ |

### Key Insight

The score-based and DDPM views are mathematically identical, but they pull your intuition in different directions. DDPM frames diffusion as *denoising* (predict the noise); score matching frames it as *learning the gradient of the data distribution*. The SDE view frames it as solving a stochastic differential equation. EDM strips away the historical baggage and presents the cleanest version: a network is trained at every noise level σ, with appropriate input/output scaling, and sampling is solving an ODE. If a 2020 paper feels obscure, restate it in EDM language — it almost always becomes obvious.

### Resources

- [Song & Ermon — Generative Modeling by Estimating Gradients (2019)](https://arxiv.org/abs/1907.05600) — the score-based foundation
- [Song et al. — Score-Based Generative Modeling through SDEs (2020)](https://arxiv.org/abs/2011.13456) — the SDE/ODE unification
- [Karras et al. — EDM (2022)](https://arxiv.org/abs/2206.00364) — the cleanest modern reformulation; *read this multiple times*
- [Ho & Salimans — Classifier-Free Guidance (2022)](https://arxiv.org/abs/2207.12598)
- [Lu et al. — DPM-Solver (2022)](https://arxiv.org/abs/2206.00927) and [DPM-Solver++ (2022)](https://arxiv.org/abs/2211.01095)
- [Yang Song — Generative Modeling by Estimating Gradients (blog)](https://yang-song.net/blog/2021/score/)
- [Sander Dieleman — Perspectives on diffusion (blog series)](https://sander.ai/2023/07/20/perspectives.html)

---

## Phase 7: Latent Diffusion and Stable Diffusion

This is where image generation went from "research demo" to "Stable Diffusion in your terminal." Master this phase and you understand the workhorse of modern T2I.

### Concepts to Learn

- **Why pixel-space diffusion stops working at high resolution**: compute is `O(H · W)`; for 512² and beyond, each forward pass becomes prohibitive
- **Latent Diffusion Models (LDM)** (Rombach et al., 2022 — the Stable Diffusion paper): train a VAE to compress 512×512×3 → 64×64×4 latents; train a U-Net diffusion model *in latent space*. Same math, ~48× less compute
- **The frozen-VAE recipe**: train the VAE once, well, then freeze it. Train the diffusion model on its latents
- **Why reconstruction quality matters more than likelihood**: a leaky VAE caps your output quality; spend serious effort training it (LPIPS + adversarial + KL regularization)
- **The famous `0.18215` scaling factor**: SD scales latents to roughly unit variance before diffusion; forgetting this breaks fine-tunes
- **Text conditioning**:
  - **CLIP text encoder** — the original choice (SD 1.x); fast but limited at 77 tokens, weaker at compositional prompts
  - **T5-XXL** — better at long prompts and adherence; Imagen, SDXL (partially), SD3, Flux
  - **Dual text encoders** (SD3, Flux): CLIP for "vibe" + T5 for "literal meaning"; concatenate or attend separately
- **The cross-attention layer**: where text tokens enter the U-Net; the same primitive used everywhere in multimodal
- **Stable Diffusion family**:
  - **SD 1.4 / 1.5** — 512², CLIP-L, U-Net; the model that launched a thousand fine-tunes
  - **SD 2.x** — OpenCLIP text encoder; less popular than 1.5
  - **SDXL** — 1024², two-stage (base + refiner), two text encoders; a meaningful quality jump
- **Cascaded vs latent-only approaches**: Imagen used cascaded super-resolution in pixel space; SD used a single latent diffusion. Latent diffusion won
- **Negative prompts**: how CFG enables "tell the model what *not* to generate"
- **img2img and inpainting**: how they fall out of the LDM formulation almost for free

### Latent Diffusion in Numbers

```
Pixel-space, 512×512:
    Each forward pass operates on 512×512×3 = 786,432 input dimensions.
    A reasonable U-Net is ~250 GFLOPs per step.
    50 steps → 12.5 TFLOPs per image.

Latent-space, 512×512 with an 8× VAE (→ 64×64×4 latent):
    Each forward pass operates on 64×64×4 = 16,384 dimensions.
    48× less data; U-Net is ~10 GFLOPs per step.
    50 steps → 0.5 TFLOPs per image.

→ 25× less compute, with comparable or better quality.
   This is why every meaningful T2I model since 2022 is latent.
```

### How Text Gets Into the U-Net

```
Prompt: "an astronaut riding a horse"
   │
   ▼
Tokenize → CLIP text encoder (and/or T5)
   │
   ▼
Sequence of (77, D) text embeddings        ← the "context"
   │
   ▼  (inside each U-Net block)
   │
   ▼
┌────────────────────────────────────────┐
│ self-attn over spatial tokens          │
│   q, k, v all come from image features │
└────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────┐
│ cross-attn from image to text          │
│   q from image features                │
│   k, v from text embeddings    ◄───────│
└────────────────────────────────────────┘
              │
              ▼
       FFN, then next block

CFG: run the whole U-Net twice per step — once with the real text emb,
once with the null (empty-string) text emb — and linearly combine.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Run SD inference | Generate images with Stable Diffusion 1.5 via `diffusers`; sweep CFG scale, samplers, step counts | ⭐⭐ |
| img2img and inpainting | Implement both from a vanilla SD inference loop; understand the noise-strength parameter | ⭐⭐⭐ |
| Train a latent DDPM | Encode CIFAR-10 with an 8× VAE; train a small U-Net in the 4×4 latent; decode and compare to pixel-space | ⭐⭐⭐⭐ |
| Train a VAE for diffusion | Train a perceptual+adversarial VAE on CelebA; verify it's a good compressor before using it for diffusion | ⭐⭐⭐⭐⭐ |
| Negative prompts study | Pick 50 prompts; compare quality with and without negative prompts ("ugly, blurry, low quality"); measure FID-CLIP and aesthetics | ⭐⭐ |
| Long-prompt test | Compare CLIP-L vs T5 conditioning on 200-token prompts; observe adherence differences | ⭐⭐⭐⭐ |
| Outpainting | Implement outpainting by inpainting an extended canvas around the original image | ⭐⭐⭐ |

### Sample Code: A Minimal Latent-Diffusion Training Step

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def encode_latent(vae, x):
    """Encode pixels to latents with a frozen VAE; SD scales by 0.18215."""
    posterior = vae.encode(x).latent_dist
    return posterior.sample() * 0.18215

def latent_ddpm_step(unet, vae, text_encoder, x_pixels, prompt_ids,
                    schedule, t, drop_cond_prob=0.1):
    # 1. Encode pixels → latents (frozen VAE)
    z0 = encode_latent(vae, x_pixels)                  # (B, 4, 64, 64) for SD-1.5

    # 2. Noise the latent
    eps = torch.randn_like(z0)
    z_t = schedule.q_sample(z0, t, eps)

    # 3. Encode text (CFG: drop the condition some of the time)
    if torch.rand(()) < drop_cond_prob:
        prompt_ids = torch.zeros_like(prompt_ids)      # treat as null/empty
    text_emb = text_encoder(prompt_ids).last_hidden_state

    # 4. Predict noise in latent space, conditioned on text via cross-attn
    eps_pred = unet(z_t, t, encoder_hidden_states=text_emb).sample

    return F.mse_loss(eps_pred, eps)
```

### Key Insight

Stable Diffusion's big-picture contribution wasn't a new generative model — it was an *engineering* recipe. Take a strong VAE (which a generative-modeling community had been quietly improving since 2016), use it as a frozen compressor, train DDPM in its latent space, condition on CLIP text embeddings via cross-attention, and release the weights publicly. None of those components were novel in isolation. The combination, plus the open release, defined the next four years of image generation. Most of what looks like "new architectures" in 2024–2026 is recombination of these same primitives.

### Resources

- [Rombach et al. — High-Resolution Image Synthesis with Latent Diffusion (2022)](https://arxiv.org/abs/2112.10752) — the Stable Diffusion paper
- [Podell et al. — SDXL (2023)](https://arxiv.org/abs/2307.01952)
- [Saharia et al. — Imagen (2022)](https://arxiv.org/abs/2205.11487) — the "T5 makes a huge difference" paper
- [`diffusers` documentation](https://huggingface.co/docs/diffusers/index) — your primary reference for SD-family inference
- [How does Stable Diffusion work? (Jay Alammar)](https://jalammar.github.io/illustrated-stable-diffusion/)

---

## Phase 8: Diffusion Transformers and Flow Matching

The U-Net era is ending. As of 2026, every meaningful new model uses a transformer backbone, and most use flow matching instead of DDPM. This phase is the current frontier.

### Concepts to Learn

- **Diffusion Transformers (DiT)** (Peebles & Xie, 2022): replace the U-Net with a pure transformer; flatten image patches to a 1D sequence of tokens. The same shift U-Net→Transformer that happened in segmentation, in detection, and in video
- **Patchification of latents**: take the VAE latent `(H', W', C)`, split into `P×P` patches, project to dim `D`. Just like ViT
- **AdaLN-Zero**: DiT's conditioning mechanism. Norm parameters (shift, scale, gate) are predicted from the conditioning vector; output gate is zero-initialized so the block is identity at start. Surprisingly robust
- **2D RoPE** (rotary positional embeddings): the modern default for spatial position; better extrapolation than learned/sinusoidal
- **DiT scaling laws**: empirically, DiT scales cleanly with parameters and FLOPs in a way U-Nets don't. This is why everyone switched
- **MMDiT (Multi-Modal DiT)**: SD3's architecture; text and image tokens share attention layers (joint attention) rather than image-attends-to-text via cross-attention. Better for compositional prompts
- **Flow Matching**: train a velocity field `v_θ(x_t, t)` such that the ODE `dx/dt = v_θ` transports noise to data
- **Rectified Flow**: a specific flow-matching parameterization where trajectories are straight lines from noise to data; allows fewer sampling steps and a cleaner objective
- **Why flow matching is replacing DDPM**: simpler loss (no schedule to tune), better-behaved at scale, near-trivial conversion of trained models to few-step sampling
- **DiT vs U-Net empirically**: DiT trains slower per FLOP but scales further; U-Net is faster per FLOP at small scales. The crossover is now well below frontier-model scale
- **Current open frontier (2024–2026)**:
  - **Flux.1** (Black Forest Labs) — currently the strongest open T2I model; MMDiT + flow matching
  - **Stable Diffusion 3.5** — Stability's MMDiT + RF release
  - **PixArt-α / PixArt-Σ** — efficient DiT recipes
  - **Hunyuan-DiT, Lumina-T2X** — competitive open releases

### A DiT Block, in Words

```
For each transformer block, condition c (time + text/class) drives 6 modulations:

    norm1 → scale_a, shift_a              ← pre-attention modulation
    attention output gated by gate_a       ← residual gate
    norm2 → scale_m, shift_m              ← pre-MLP modulation
    MLP output gated by gate_m             ← residual gate

The (shift, scale, gate) triple is "AdaLN-Zero":
    - shift  : channel-wise additive
    - scale  : channel-wise multiplicative
    - gate   : output magnitude; zero-initialized so block is identity at start

In MMDiT (SD3), text and image tokens are concatenated, but each token type
has its own AdaLN parameters and its own MLP. The attention is joint.
```

### The Flow-Matching Objective in One Equation

```
Sample t ~ U(0, 1),   ε ~ N(0, I).

Linear interpolation between data and noise:
    x_t = (1 - t) · x_0 + t · ε

The "target velocity" along this path is:
    v_true = ε - x_0

Train v_θ to match it:
    L_FM = E [ || v_θ(x_t, t) - v_true ||² ]

At sample time, solve the ODE dx/dt = v_θ(x, t) from t=1 (noise) to t=0 (data),
typically with Euler or Heun, using 10–50 steps.

Rectified flow's idea: after training, "re-flow" by training a second model
on the straight-line paths from x_T to x_0 produced by the first. Trajectories
get straighter, fewer steps are needed at inference.
```

### Sample Code: A Minimal DiT Block (AdaLN-Zero)

```python
import torch
import torch.nn as nn

class DiTBlock(nn.Module):
    """A transformer block with AdaLN-Zero conditioning, à la Peebles & Xie."""
    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # Six modulation parameters: (shift, scale, gate) × {attn, mlp}
        # Initialized to zero so the block is identity at init (the "Zero").
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        # c: conditioning vector (timestep + class/text emb), shape (B, dim)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = \
            self.ada(c).chunk(6, dim=-1)

        h = self.norm1(x) * (1 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_a.unsqueeze(1) * h

        h = self.norm2(x) * (1 + scale_m.unsqueeze(1)) + shift_m.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_m.unsqueeze(1) * h
        return x
```

### Sample Code: A Flow-Matching Training Step

```python
import torch
import torch.nn.functional as F

def flow_matching_step(model, x_0, cond=None):
    # Sample noise and timestep
    eps = torch.randn_like(x_0)
    t = torch.rand(x_0.size(0), device=x_0.device)        # uniform on (0, 1)
    t_view = t.view(-1, 1, 1, 1)

    # Linear interpolation between data and noise
    x_t = (1 - t_view) * x_0 + t_view * eps

    # Target velocity
    v_true = eps - x_0

    # Predict velocity, regress
    v_pred = model(x_t, t, cond)
    return F.mse_loss(v_pred, v_true)
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Implement DiT-S/2 | Small DiT (12 blocks, dim 384), patch size 2; train class-conditional on CIFAR-10; compare to your U-Net baseline | ⭐⭐⭐⭐ |
| 2D RoPE for DiT | Add 2D rotary positional embeddings; verify quality improves vs learned positions | ⭐⭐⭐ |
| Rectified flow from scratch | Re-train the above with the flow-matching objective; observe convergence and few-step sampling | ⭐⭐⭐⭐ |
| MMDiT block | Implement an SD3-style joint text-image attention block; train a tiny version on a captioned dataset | ⭐⭐⭐⭐⭐ |
| Re-flow | Take a trained rectified-flow model; generate (noise, sample) pairs; train a second model on those straight-line paths | ⭐⭐⭐⭐⭐ |
| Compare DiT and U-Net scaling | Train DiT-S, DiT-B, DiT-L on the same dataset; plot FID vs compute; observe DiT's slope advantage | ⭐⭐⭐⭐⭐ |

### Key Insight

Two architectural shifts have happened almost simultaneously: U-Net → Transformer, and DDPM → Flow Matching. Neither is strictly necessary for the other, but they showed up together in SD3, Flux, and most 2024+ video models. The case for transformers is the same case that won in NLP — they scale predictably and have clean primitives. The case for flow matching is that it removes a layer of historical baggage (variance schedules, noise schedules, the choice of `ε`-vs-`x_0`-vs-`v` prediction) and gives you a single, clean regression target with no hyperparameters to sweep. If you're starting a new image-gen project in 2026, "DiT + flow matching" is the default; you need a reason to deviate from it.

### Resources

- [Peebles & Xie — DiT (2022)](https://arxiv.org/abs/2212.09748) — the foundational paper
- [Esser et al. — Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (2024)](https://arxiv.org/abs/2403.03206) — the SD3 paper, definitive treatment of MMDiT
- [Liu et al. — Rectified Flow (2022)](https://arxiv.org/abs/2209.03003)
- [Lipman et al. — Flow Matching for Generative Modeling (2022)](https://arxiv.org/abs/2210.02747)
- [Chen et al. — PixArt-α (2023)](https://arxiv.org/abs/2310.00426) — efficient DiT for T2I
- [Flux.1 (Black Forest Labs)](https://blackforestlabs.ai/) — current open frontier

---

## Phase 9: Conditioning, Control, and Personalization

A great unconditional generator is interesting; a great *controllable* generator is useful. This phase is about everything that wraps the core model so you get the picture you actually wanted.

### Concepts to Learn

- **Text conditioning revisited**: long-prompt handling, prompt weighting, attention-region steering
- **Image conditioning**:
  - **img2img** — start denoising from a noised real image instead of from pure noise
  - **Inpainting** — mask a region; denoise only within the mask; condition on the surrounding pixels
  - **Outpainting** — inpainting at the edges of an extended canvas
- **ControlNet** (Zhang et al., 2023): add a parallel branch that conditions on auxiliary signals (depth, edges, pose, segmentation). The breakthrough that made diffusion *controllable*
- **Zero-convolutions**: ControlNet's init trick — gradient flow but identity at start, so a pretrained model isn't disturbed
- **T2I-Adapter**: lighter-weight alternative to ControlNet; smaller, simpler, similar idea
- **IP-Adapter**: image-prompt conditioning via a small attention adapter; subject reference at inference, no fine-tuning needed
- **InstantID, PhotoMaker**: identity-preserving generation from one or few reference photos
- **Personalization via fine-tuning**:
  - **DreamBooth** (Ruiz et al., 2022) — fine-tune the whole model on 3–5 images of a subject; quality is high but model is expensive
  - **Textual Inversion** (Gal et al., 2022) — learn a new "word" embedding for the subject; tiny but limited
  - **LoRA** (Hu et al., 2021) — low-rank adapters; the standard middle ground; small files, fast training, easy to share
  - **LyCORIS, DoRA, LoHa** — the LoRA family
- **Editing real images**:
  - **SDEdit** — partial-noise editing; the simplest method
  - **DDIM inversion** — invert real images into the diffusion latent space, then edit
  - **Prompt-to-Prompt, Null-Text Inversion** — preserve structure while changing content
  - **InstructPix2Pix** — instruction-tuned editing; one-shot inference
- **Compositional control**:
  - **MultiDiffusion / SyncDiffusion** — run diffusion on overlapping regions; tile arbitrary aspect ratios
  - **RegionalPrompter** — different prompts in different regions
- **Style transfer with diffusion**: StyleAligned, B-LoRA, IP-Adapter for style

### A Taxonomy of Control Surfaces

```
INPUT                        →  TASK                       EXAMPLES
─────────────────────────       ────────────────────────   ──────────────────
text                         →  T2I                        SD, Flux, Imagen
text + reference image       →  identity-preserved gen     IP-Adapter, InstantID
text + structure (depth/edge)→  controlled gen             ControlNet
text + region mask           →  inpainting/outpainting     SD inpaint
text + real image            →  edit (style/object)        SDEdit, P2P
text + instruction + image   →  instruction edit           InstructPix2Pix
3–5 subject photos           →  personalization            DreamBooth, LoRA
text + multiple regions      →  compositional gen          RegionalPrompter
```

### Sample Code: A Tiny ControlNet-Style Zero-Init Branch

```python
import torch
import torch.nn as nn

class ZeroConv(nn.Conv2d):
    """A conv whose weights and bias are zero-initialized.
    Lets a parallel branch contribute nothing at init, then learn from scratch."""
    def __init__(self, in_c, out_c):
        super().__init__(in_c, out_c, kernel_size=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class ControlBranch(nn.Module):
    """Mirror an existing encoder; feed the conditioning signal in;
    add zero-conv outputs to the corresponding skip connections of the main U-Net."""
    def __init__(self, encoder_clone, dims):
        super().__init__()
        self.encoder = encoder_clone                  # a deep copy of the U-Net's encoder
        self.zero_convs = nn.ModuleList(ZeroConv(d, d) for d in dims)

    def forward(self, control, t, cond):
        # control: e.g., a Canny edge map matching the input image
        feats = self.encoder(control, t, cond)        # list of (B, C, H, W) skips
        return [zc(f) for zc, f in zip(self.zero_convs, feats)]
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| img2img and inpainting | Implement both from a vanilla SD inference loop; tune noise strength | ⭐⭐⭐ |
| LoRA fine-tune | Take a pretrained SD checkpoint; fine-tune a LoRA on 20 images of a custom subject; verify it works | ⭐⭐⭐ |
| DreamBooth | Full fine-tune on a small subject set; compare quality and parameter count to LoRA | ⭐⭐⭐⭐ |
| Textual Inversion | Train just a new token embedding for the same subject; compare to LoRA | ⭐⭐⭐ |
| ControlNet from scratch | Implement the ControlNet "zero-conv" branch; condition on Canny edges; train on a small dataset | ⭐⭐⭐⭐⭐ |
| InstructPix2Pix-style data | Generate a synthetic instruction-edit dataset using a pretrained T2I + GPT; fine-tune a small editor | ⭐⭐⭐⭐⭐ |
| DDIM inversion + edit | Invert a real photo into latent noise; modify the prompt; denoise; observe structure preservation | ⭐⭐⭐⭐ |
| Style LoRA | Train a LoRA on 30 images in a coherent style; verify the style transfers to unseen prompts | ⭐⭐⭐ |

### Key Insight

In image generation, ControlNet and its successors made the difference between "generate something cool" and "generate exactly what I want." Before ControlNet, T2I models were impressive but unreliable production tools — you couldn't fix the pose, fix the composition, or fix the identity. After ControlNet (and LoRA, and IP-Adapter), they became actual creative tools. The frontier in 2026 is no longer "make the base model bigger"; it's "make the control surfaces more precise, more composable, and more interactive." Whoever ships the next great control primitive — at the right level of abstraction — defines the next generation of commercial tools.

### Resources

- [Zhang et al. — ControlNet (2023)](https://arxiv.org/abs/2302.05543)
- [Mou et al. — T2I-Adapter (2023)](https://arxiv.org/abs/2302.08453)
- [Ye et al. — IP-Adapter (2023)](https://arxiv.org/abs/2308.06721)
- [Ruiz et al. — DreamBooth (2022)](https://arxiv.org/abs/2208.12242)
- [Gal et al. — Textual Inversion (2022)](https://arxiv.org/abs/2208.01618)
- [Hu et al. — LoRA (2021)](https://arxiv.org/abs/2106.09685)
- [Meng et al. — SDEdit (2021)](https://arxiv.org/abs/2108.01073)
- [Brooks et al. — InstructPix2Pix (2022)](https://arxiv.org/abs/2211.09800)
- [Hertz et al. — Prompt-to-Prompt (2022)](https://arxiv.org/abs/2208.01626)
- [Bar-Tal et al. — MultiDiffusion (2023)](https://arxiv.org/abs/2302.08113)

---

## Phase 10: Training at Scale, Distillation, Evaluation, and Frontier Topics

The operational reality of image generation: data, compute, eval, fast inference, and what's still open.

### Training at Scale

- **Data sources**:
  - Public: **LAION-5B, LAION-2B-en**, **COYO-700M**, **DataComp**, **OpenImages**, **JourneyDB** (Midjourney-recreated captions)
  - Proprietary: most strong commercial models train on licensed stock-image libraries (Getty, Shutterstock, Adobe Stock) and curated web data
- **Data filtering**: deduplication (perceptual hash), NSFW filtering, CSAM detection (mandatory, not optional), aesthetic-score filtering, OCR-text filtering, watermark detection
- **Synthetic captions** — the single highest-leverage data trick. Recaptioning web images with a strong VLM (LLaVA, Qwen2-VL, GPT-4o) dramatically improves prompt adherence. This is the secret behind DALL-E 3's compositional ability
- **Aspect-ratio bucketing**: train on multiple aspect ratios together for variable-resolution inference; don't center-crop everything to square
- **Curriculum**: start at low resolution and small captions, scale up gradually
- **Compute budgets**:
  - SD-1.5 was trained on ~256 A100s for several weeks
  - SDXL: noticeably more
  - Flux / Imagen 3 / DALL-E 3 / Midjourney v6: frontier scale, undisclosed but on the order of 10²³–10²⁴ FLOPs

### Distillation and Few-Step Models

The other axis of scaling: making sampling cheaper, not training.

- **Progressive distillation** (Salimans & Ho, 2022): student does 2 steps for every 4 of the teacher; iterate to get few-step models
- **Consistency Models** (Song et al., 2023): a different parameterization that allows 1–4-step sampling directly
- **Latent Consistency Models (LCM)**: consistency models in latent space; the most practical version for SD-style stacks
- **SDXL Turbo, SD3 Turbo, Flux Schnell**: production 1–4-step models. SDXL Turbo uses Adversarial Diffusion Distillation (ADD); Flux Schnell uses a different recipe
- **DMD (Distribution Matching Distillation)** (Yin et al., 2024): another strong few-step recipe
- **Why this matters**: a 1-second image-gen latency unlocks interactive use cases (real-time previews, in-editor generation) that 10-second latency does not

### Evaluation

The evaluation problem in image generation is bad, and people keep papering over it. Knowing which numbers to trust is its own skill.

- **Automatic metrics**:
  - **FID** — the standard; criticized for sensitivity to feature extractor and sample count; use `clean-fid` for reproducibility
  - **CLIP-FID** — FID computed with CLIP features; less biased toward photorealism
  - **CLIPScore** — text-image alignment
  - **Aesthetic predictors** (LAION-Aesthetic, MPS, ImageReward) — learned proxies for "pretty"
  - **DINOv2-FID** — recent variant, more robust
- **Human evaluation** — pairwise comparison, win rates; still the gold standard, especially for prompt adherence
- **LLM-as-judge** — using GPT-4V or Claude to grade image-text alignment; biased but useful
- **Benchmarks**:
  - **GenEval** — compositional prompt adherence
  - **T2I-CompBench** — compositional reasoning
  - **DPG-Bench** — dense-prompt generation
  - **PartiPrompts** — Parti's curated prompt set; common reference
- **Hallucination / mode failures**: text rendering, hand anatomy, finger counts, complex spatial relations — known weak spots

### Frontier Topics

- **Real-time generation**: 1-step diffusion, consistency models, autoregressive caching. Sub-100-ms latency is now possible
- **Text rendering in images**: long the worst failure mode; mostly solved as of Imagen 3 / Flux / Ideogram 2.0 via dedicated training data and architectural tricks
- **3D-aware image generation**: HoloDiffusion, viewpoint-controllable diffusion, generative NeRFs; the bridge to true 3D generation
- **High-resolution and ultra-high-resolution**: 4K and beyond; cascaded approaches, patch-based generation, attention sparsity
- **Compositional and structured generation**: making models reliably count, position, and relate multiple objects
- **Safety**: watermarking (e.g., SynthID, Tree-Ring), deepfake detection, dataset poisoning attacks (Glaze, Nightshade), consent and provenance
- **Unified image gen + understanding**: models that both generate and recognize (Show-o, Emu3, Transfusion). The natural next step toward GPT-4o-style omni models
- **Editing and re-rendering** at production quality: precise local edits, identity preservation through edits, multi-step iterative editing

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Mini LAION pipeline | Take 1M LAION URLs, download, filter with CLIP, dedup, recaption with a small VLM — produce a clean shard | ⭐⭐⭐⭐ |
| Caption ablation | Train two small T2I models: one on original alt-text, one on recaptioned text; compare downstream | ⭐⭐⭐⭐⭐ |
| Aspect-ratio bucketing | Implement bucketed batching for variable aspect ratios; observe quality improvement on portrait/wide test sets | ⭐⭐⭐ |
| Consistency distillation | Distill a 50-step SD model into a 4-step LCM student; measure quality loss | ⭐⭐⭐⭐⭐ |
| Adversarial Diffusion Distillation | Implement ADD (SDXL Turbo's recipe); compare to LCM | ⭐⭐⭐⭐⭐ |
| Watermarking | Add invisible watermarking to your model's outputs; verify with a detector | ⭐⭐⭐⭐ |
| GenEval run | Evaluate an open T2I model on GenEval; document failure modes | ⭐⭐ |
| Human-correlated eval | For 100 outputs, get 3 human ratings and 3 LLM-as-judge ratings; measure agreement | ⭐⭐⭐ |
| Text-rendering probe | Construct 200 prompts that include rendered text ("a sign that says '...'"); evaluate open models | ⭐⭐ |

### Key Insight

Image generation in 2026 is converging *fast*. The architectural recipe (VAE + DiT + flow matching + CFG + dual text encoders) is consensus across nearly every strong open and closed model. The interesting differences are *data* (curation, recaptioning, licensing), *scale* (the obvious one), and *control surfaces* (where commercial differentiation lives). The compute frontier is no longer impossibly far — training a Flux-class model is within reach of well-funded organizations — but obtaining the licensed and recaptioned data to train it on is harder than it used to be. If you're entering the field, the highest-leverage skills are not architecture, they're data engineering, evaluation methodology, distillation for speed, and control-surface design.

### Resources

- [LAION-5B paper](https://arxiv.org/abs/2210.08402)
- [DataComp paper](https://arxiv.org/abs/2304.14108) — careful study of data curation
- [Betker et al. — DALL-E 3 technical report (2023)](https://cdn.openai.com/papers/dall-e-3.pdf) — recaptioning recipe
- [Salimans & Ho — Progressive Distillation (2022)](https://arxiv.org/abs/2202.00512)
- [Song et al. — Consistency Models (2023)](https://arxiv.org/abs/2303.01469)
- [Luo et al. — Latent Consistency Models (2023)](https://arxiv.org/abs/2310.04378)
- [Sauer et al. — Adversarial Diffusion Distillation (SDXL Turbo, 2023)](https://arxiv.org/abs/2311.17042)
- [Yin et al. — DMD (2023)](https://arxiv.org/abs/2311.18828)
- [Ghosh et al. — GenEval (2023)](https://arxiv.org/abs/2310.11513)
- [`clean-fid` (Parmar et al.)](https://github.com/GaParmar/clean-fid) — properly-computed FID
- [SynthID for images (DeepMind)](https://deepmind.google/technologies/synthid/)

---

## Suggested Timeline

| Phase | Duration | Outcome |
|-------|----------|---------|
| 0. Prerequisites | 0–2 weeks | PyTorch + transformer + probability basics solid |
| 1. Foundations | 1 week | Comfortable with the model-family taxonomy and FID |
| 2. VAEs | 1–2 weeks | Trained a VAE; understand the ELBO and posterior collapse |
| 3. Discrete latents | 1–2 weeks | Trained a VQ-VAE and a VQ-GAN; tokenization as a primitive |
| 4. GANs | 1–2 weeks | Trained DCGAN and WGAN-GP; watched mode collapse firsthand |
| 5. Diffusion foundations | 2–3 weeks | DDPM on CIFAR-10 trained from scratch; DDIM sampling |
| 6. Score / EDM / CFG | 2 weeks | Reformulated in σ-space; CFG implemented; multiple samplers compared |
| 7. Latent diffusion | 2 weeks | Ran SD inference deeply; trained a small latent DDPM |
| 8. DiT + flow matching | 2–3 weeks | Implemented DiT and rectified flow; compared to U-Net |
| 9. Conditioning + control | 2–3 weeks | Trained a LoRA, a ControlNet, and at least one personalization recipe |
| 10. Scale + distillation + eval | Ongoing | Real benchmark evaluation; data pipeline understood; few-step distillation tried |

**Total to "comfortable practitioner":** ~3–5 months of focused study. From here, [Video Generation](../video-generation/) and [Multimodal Learning](../multimodal-learning/) are natural next steps.

---

## Key Advice

1. **Skip pixel-space at 256² and above.** Once you understand DDPM on CIFAR, move to latent diffusion. Pixel-space high-res is a research curiosity, not a practical tool.
2. **Train a small DDPM from scratch at least once.** Reading the paper is not enough. The loop is a hundred lines; the lessons last forever.
3. **The VAE matters as much as the diffusion model.** SD's VAE is doing real work; a bad VAE caps everything downstream. If you're rolling your own, budget serious time here.
4. **CFG scale is a hyperparameter, not a constant.** 1.0 is no guidance; 7.5 is the SD default; 12+ is "looks AI". Sweep it.
5. **Don't trust FID alone.** Sample by hand. Look at the images. FID can be gamed and is sensitive to feature-extractor choice and sample count.
6. **`bf16` everywhere on Ampere+.** Same advice as elsewhere in the stack. `float16` GradScalers are unnecessary friction.
7. **EMA your weights.** Exponential moving average of generator weights is standard practice; samples from the EMA are noticeably better than samples from the live weights.
8. **The samplers matter less than you'd think.** DDIM with 50 steps is fine. DPM-Solver++ with 20 steps is also fine. Don't optimize this before you've fixed the model.
9. **Synthetic captions are a superpower.** Recaptioning your training images with a strong VLM is the single highest-leverage data trick — the secret behind DALL-E 3's prompt adherence.
10. **Read EDM (Karras 2022) until it makes sense.** The σ-space reformulation makes everything cleaner. Modern papers assume you've internalized it.

---

## Common Pitfalls to Avoid

- ❌ Training a VAE with only pixel MSE and being surprised the reconstructions are blurry
- ❌ Forgetting to scale the SD latents by `0.18215` when plugging in a custom VAE
- ❌ Using `float16` (with a GradScaler) instead of `bf16` for diffusion training — silent NaNs are common
- ❌ Skipping classifier-free guidance training and being disappointed by conditional sample quality
- ❌ Reporting FID without specifying number of samples and the Inception feature extractor version
- ❌ Loading a pretrained checkpoint with `strict=True` and `torch.load` defaults, in 2026 — use `safetensors`
- ❌ Tuning CFG scale before fixing the schedule, the encoder, and the training recipe
- ❌ Treating GANs as a starting point in 2026 — they're a historical chapter; diffusion is the default
- ❌ Using a CLIP-L text encoder for long-form prompts beyond ~77 tokens — switch to T5 or a dual encoder
- ❌ Training a U-Net at high resolution in pixel space instead of moving to latents
- ❌ Believing FID is a good proxy for prompt adherence; use GenEval or human eval for that
- ❌ Training without dropping the condition 10% of the time; CFG won't work later

---

## Additional Resources

### Books and Long-Form Reading
- [Lilian Weng — What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [The Annotated Diffusion Model (Hugging Face)](https://huggingface.co/blog/annotated-diffusion)
- [Sander Dieleman — Diffusion blog series](https://sander.ai/) — read the whole archive
- [Yang Song — Generative Modeling by Estimating Gradients (blog)](https://yang-song.net/blog/2021/score/)
- [Goodfellow, Bengio, Courville — Deep Learning, Ch. 20](https://www.deeplearningbook.org/contents/generative_models.html) — dated but foundational

### Key Papers, Chronologically
| Year | Paper | Contribution |
|------|-------|-------------|
| 2013 | [VAE](https://arxiv.org/abs/1312.6114) | Variational lower bound; reparam trick |
| 2014 | [GAN](https://arxiv.org/abs/1406.2661) | The adversarial game |
| 2015 | [DCGAN](https://arxiv.org/abs/1511.06434) | First reliable GAN architecture |
| 2017 | [WGAN-GP](https://arxiv.org/abs/1704.00028) | Stability via gradient penalty |
| 2017 | [PixelCNN++](https://arxiv.org/abs/1701.05517) | Strong autoregressive baseline |
| 2017 | [VQ-VAE](https://arxiv.org/abs/1711.00937) | Discrete latents |
| 2018 | [StyleGAN](https://arxiv.org/abs/1812.04948) | Style-based generator |
| 2019 | [Song & Ermon — Score Matching](https://arxiv.org/abs/1907.05600) | Score-based generative modeling |
| 2020 | [DDPM](https://arxiv.org/abs/2006.11239) | Modern diffusion |
| 2020 | [DDIM](https://arxiv.org/abs/2010.02502) | Deterministic accelerated sampling |
| 2020 | [Score SDE](https://arxiv.org/abs/2011.13456) | SDE unification |
| 2020 | [VQ-GAN](https://arxiv.org/abs/2012.09841) | Perceptual + adversarial for VQ; SD's VAE recipe |
| 2021 | [Improved DDPM](https://arxiv.org/abs/2102.09672) | Cosine schedule, learned variance |
| 2021 | [Diffusion Beats GANs](https://arxiv.org/abs/2105.05233) | Classifier guidance; FID milestone |
| 2021 | [LoRA](https://arxiv.org/abs/2106.09685) | Low-rank adaptation |
| 2022 | [Latent Diffusion](https://arxiv.org/abs/2112.10752) | The Stable Diffusion paper |
| 2022 | [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) | The CFG trick |
| 2022 | [Imagen](https://arxiv.org/abs/2205.11487) | T5 text conditioning matters |
| 2022 | [EDM (Karras)](https://arxiv.org/abs/2206.00364) | σ-space, modern formulation |
| 2022 | [DiT](https://arxiv.org/abs/2212.09748) | Transformer backbone for diffusion |
| 2022 | [Rectified Flow](https://arxiv.org/abs/2209.03003) / [Flow Matching](https://arxiv.org/abs/2210.02747) | Cleaner objective |
| 2022 | [DreamBooth](https://arxiv.org/abs/2208.12242) / [Textual Inversion](https://arxiv.org/abs/2208.01618) | Subject personalization |
| 2023 | [ControlNet](https://arxiv.org/abs/2302.05543) | The controllability moment |
| 2023 | [SDXL](https://arxiv.org/abs/2307.01952) | 1024² latent diffusion |
| 2023 | [Consistency Models](https://arxiv.org/abs/2303.01469) | Few-step sampling |
| 2023 | [DALL-E 3 (technical report)](https://cdn.openai.com/papers/dall-e-3.pdf) | Synthetic-caption recipe |
| 2024 | [SD3 (MMDiT)](https://arxiv.org/abs/2403.03206) | Joint text-image transformer + RF |
| 2024 | [Flux.1](https://blackforestlabs.ai/) | Current open frontier |

### Tools You Should Know
- **`diffusers` (Hugging Face)** — inference and prototyping for nearly every open T2I model
- **`accelerate`** — multi-GPU training without writing the loop yourself
- **`safetensors`** — the file format you should be using
- **`peft`** — for LoRA and other parameter-efficient fine-tuning
- **`xformers` / FlashAttention** — fast attention kernels; the difference between affordable and unaffordable
- **`ComfyUI` / `Automatic1111`** — for rapid prototyping with open models and visual graph pipelines
- **`clean-fid`** — properly-computed FID, the right way
- **`webdataset`** — streaming large image-text data

### Communities
- [Hugging Face forums and Discord](https://discuss.huggingface.co/)
- [r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/) — practical, open-source-focused; moves fast
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) — research-oriented
- Twitter/X — diffusion research moves on Twitter faster than anywhere else

---

## Quick Start Checklist

- [ ] Can explain why max-likelihood and "good samples" pull in different directions
- [ ] Have trained a VAE and observed posterior collapse at high β
- [ ] Have trained a VQ-VAE and understand the straight-through estimator
- [ ] Have trained a DCGAN and seen mode collapse firsthand
- [ ] Have derived the closed-form `x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε`
- [ ] Have trained a DDPM on CIFAR-10 to FID < 20
- [ ] Have implemented DDIM sampling and verified it matches DDPM in fewer steps
- [ ] Have re-derived a diffusion model in EDM σ-space
- [ ] Have implemented classifier-free guidance and swept the scale
- [ ] Have run Stable Diffusion inference and inspected its latent space
- [ ] Have implemented a DiT block and verified it trains
- [ ] Have implemented flow matching and compared convergence to DDPM
- [ ] Have trained at least one LoRA or ControlNet on a custom task
- [ ] Have distilled a multi-step model into a few-step student

---

## Glossary

| Term | Definition |
|------|------------|
| **AdaLN-Zero** | DiT's conditioning mechanism: layer norm modulated by shift/scale/gate, all initialized to zero |
| **ADD (Adversarial Diffusion Distillation)** | SDXL Turbo's recipe: distill a multi-step diffusion model into a 1–4-step student using a discriminator loss |
| **bpd (bits per dimension)** | Standard likelihood metric for image models; `-log₂ p(x) / D` |
| **CFG (classifier-free guidance)** | Inference trick: combine conditional and unconditional model outputs to amplify conditioning |
| **Consistency model** | A diffusion-derived model that samples in 1–4 steps via consistency distillation |
| **ControlNet** | Architecture that adds an auxiliary conditioning branch (depth, pose, edges, ...) to a frozen diffusion model |
| **Cross-attention** | Attention where queries come from one modality (image) and keys/values from another (text) |
| **DDIM** | Deterministic, accelerated sampler for diffusion models |
| **DDPM** | Denoising Diffusion Probabilistic Models — the foundational 2020 paper and training recipe |
| **DiT** | Diffusion Transformer — Peebles & Xie's transformer-based diffusion backbone |
| **DreamBooth** | Fine-tuning recipe for subject personalization; updates the whole model on a few subject images |
| **EDM** | Karras et al. 2022 — a reformulation of diffusion in σ-space with clean preconditioning |
| **ELBO** | Evidence Lower Bound — the variational objective trained by VAEs |
| **EMA weights** | Exponential moving average of model weights; samples better than the live weights |
| **FID** | Fréchet Inception Distance — the standard sample-quality metric for image generation |
| **Flow matching** | Training a velocity field that transports noise to data via an ODE; modern alternative to DDPM |
| **FSQ** | Finite Scalar Quantization — codebook-free discrete tokenization |
| **LCM** | Latent Consistency Model — consistency-distilled few-step latent diffusion |
| **LDM** | Latent Diffusion Model — diffusion in the latent space of a VAE (i.e., Stable Diffusion) |
| **LoRA** | Low-Rank Adaptation — parameter-efficient fine-tuning via low-rank weight deltas |
| **MMDiT** | Multi-Modal Diffusion Transformer — joint text+image attention layers, used in SD3 and Flux |
| **Mode collapse** | GAN failure mode: generator produces few distinct outputs |
| **Perceptual loss (LPIPS)** | Loss computed in the feature space of a pretrained classifier; sharper than pixel MSE |
| **Posterior collapse** | VAE failure mode: encoder collapses to the prior; latent carries no information |
| **Probability flow ODE** | The deterministic ODE equivalent of the reverse-time diffusion SDE |
| **Rectified flow** | A specific flow-matching parameterization with straight-line trajectories |
| **Reparameterization trick** | `z = μ + σ · ε` — lets gradients flow through a random sample |
| **Score** | `∇_x log p(x)` — diffusion training implicitly learns this |
| **U-Net** | Encoder-decoder architecture with skip connections; the standard diffusion backbone before DiT |
| **VAE (variational autoencoder)** | Encoder/decoder pair trained on the ELBO; used as a compressor in latent diffusion |
| **VP / VE SDE** | Variance-Preserving / Variance-Exploding — the two SDE families for diffusion |
| **VQ-VAE** | Vector-quantized VAE — discrete latent codes from a learned codebook |
| **VQ-GAN** | VQ-VAE trained with perceptual + adversarial losses; SD's VAE recipe descends from this |
| **σ-schedule (Karras)** | The EDM reformulation: parameterize diffusion by noise standard deviation σ rather than discrete timestep |
| **Zero-conv** | A 1×1 convolution with zero-initialized weights and bias; used by ControlNet to add a branch without disturbing init |

---

## License

MIT License. See the [LICENSE](https://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
