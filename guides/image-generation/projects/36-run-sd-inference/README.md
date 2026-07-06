# Run SD Inference

## ELI5 (Explain Like I'm 5)

- **The Big Idea:** Stable Diffusion 1.5 is a powerful model that can generate high-quality images from text. This project runs the model using the `diffusers` library and lets you experiment with the three main knobs: how many steps to run (speed vs quality), which sampler to use (how it solves the math), and the guidance scale (how closely it follows your words).
- **Analogy:** Think of Stable Diffusion as a high-tech camera. The text prompt is the scene you want to shoot, the sampler is the lens type, the step count is the exposure time, and the guidance scale is the focus contrast. Changing these settings changes the style and sharpness of the photo.
- **Example:** You type "a cat astronaut." With a guidance scale of `1.0`, you get a blurry astronaut shape. With `7.5`, you get a sharp, cute cat in a space suit. With `20.0`, the colors become cartoonishly bright and high-contrast because the model is trying too hard to follow the prompt.


## Key Insight

[Stable Diffusion](/shared/glossary/#stable-diffusion) 1.5 turned text-to-image generation into something you can run on a single consumer GPU, and the `diffusers` library wraps the whole pipeline — text encoder, [U-Net](/shared/glossary/#u-net), and [VAE](/shared/glossary/#vae) decoder — behind a few lines of code. This project builds intuition for the three knobs that matter most at inference: the [classifier-free guidance (CFG)](/shared/glossary/#cfg-classifier-free-guidance) scale (how hard the model is pushed toward your prompt), the sampler (the [ODE/SDE](/shared/glossary/#ode) solver that takes each denoising step, e.g. [DDIM](/shared/glossary/#ddim) or [DPM-Solver++](/shared/glossary/#dpm-solver)), and the number of steps (more steps = slower but usually cleaner). Sweeping each one while holding the others fixed exposes the trade-offs — high CFG sharpens prompt adherence but oversaturates, while a [higher-order sampler](/shared/glossary/#higher-order-sampler) reaches good quality in far fewer steps — all without touching the model weights.

## What's in this directory

| File | Role |
|------|------|
| `run_sd_inference.py` | Loads the pipeline and runs the three sweeps, one contact sheet each |
| `contact.py` | Small helper that pastes labeled images into one sheet (reused by the [img2img and inpainting](../37-img2img-and-inpainting/README.md) and [Negative prompts study](../40-negative-prompts-study/README.md) projects–42) |

**About the model.** The recorded run uses `segmind/tiny-sd` — a
knowledge-distilled Stable Diffusion 1.5 with the identical architecture,
latent space, text encoder, and `diffusers` API, at roughly half the U-Net —
generated at 384×384 so all eleven images finish in about six minutes on a
CPU. On a GPU, set `MODEL_ID` to SD 1.5 proper and `SIZE = 512`; not one
other line changes. Every observation below transfers, because the knobs
being swept belong to the *sampling machinery*, not to any particular
checkpoint.

```bash
python run_sd_inference.py       # ~6 min on a multicore CPU
```

Note what the three sweeps really are: everything in this pipeline is
machinery you have already built by hand — the CFG extrapolation is project
32, the samplers are the [Higher-order sampler](../31-higher-order-sampler/README.md) project's solvers on the [Probability flow ODE](../34-probability-flow-ode/README.md) project's ODE, and the VAE
encode/decode bracket is the [Train a latent DDPM](../38-train-a-latent-ddpm/README.md) and [Train a VAE for diffusion](../39-train-a-vae-for-diffusion/README.md) projects. This project is where those pieces
meet a 900M-parameter production model.

## Results

**CFG sweep** (DPM++ 2M, 15 steps, same seed). At `CFG 1` — no guidance —
the image is muddy and only loosely on-prompt; `3` is coherent but soft;
`7.5` (the community default) is crisp and saturated; `12` pushes contrast
and color to the edge of poster-like:

![CFG scale sweep](outputs/cfg_sweep.png)

**Step-count sweep** (DPM++ 2M, CFG 7.5). Three steps is recognizably the
right scene with smeared detail, and the gap from 12 to 25 is subtle — the
higher-order-solver lesson from the [Higher-order sampler](../31-higher-order-sampler/README.md) project operating at full scale:

![Step count sweep](outputs/steps_sweep.png)

**Sampler comparison at 8 steps** (CFG 7.5). At generous step counts all
good samplers agree; starving them to 8 steps is where they differentiate.
DPM++ 2M holds together best at this budget, DDIM is the softest, Euler
in between — same ranking the solver-error curves of the [Higher-order sampler](../31-higher-order-sampler/README.md) project predict:

![Sampler comparison](outputs/sampler_sweep.png)

## Things to try

- Fix everything and vary only the seed: same knobs, wildly different
  compositions — the noise is the composition.
- Sweep CFG at 3 steps. Guidance and step count interact: high CFG needs
  enough steps to absorb the push it applies.
- Print `pipe.unet.config.cross_attention_dim` and trace where the text
  embeddings enter — then look at the [MMDiT block](../46-mmdit-block/README.md) project for how the
  same wiring looks in current-generation models.
