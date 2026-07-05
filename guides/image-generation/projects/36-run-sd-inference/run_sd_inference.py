"""Run Stable Diffusion inference and sweep the three knobs that matter:
CFG scale, step count, and sampler.

Model note: the recorded run uses `segmind/tiny-sd` — a knowledge-distilled
Stable Diffusion 1.5 (identical architecture and API, roughly half the
U-Net) — at 384x384 so the whole sweep finishes in minutes on a CPU. On a
GPU, change MODEL_ID to "stable-diffusion-v1-5/stable-diffusion-v1-5" and
SIZE to 512: every line below works unchanged. That interchangeability IS
the lesson: the pipeline is the same latent-diffusion machine either way.

Run:
    python run_sd_inference.py            # ~6 min on a multicore CPU
"""

import argparse
import time
from pathlib import Path

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
)

from contact import contact_sheet

HERE = Path(__file__).resolve().parent
MODEL_ID = "segmind/tiny-sd"
SIZE = 384
PROMPT = "a watercolor painting of a lighthouse on a rocky coast at sunset"


def load_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def gen(pipe, seed=0, **kwargs):
    t0 = time.time()
    img = pipe(PROMPT, height=SIZE, width=SIZE,
               generator=torch.Generator().manual_seed(seed), **kwargs).images[0]
    return img, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipe()

    # 1) CFG sweep — fixed sampler (DPM++) and steps.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    images, labels = [], []
    for cfg in (1.0, 3.0, 7.5, 12.0):
        img, s = gen(pipe, num_inference_steps=15, guidance_scale=cfg)
        images.append(img)
        labels.append(f"CFG {cfg}")
        print(f"CFG {cfg}: {s:.0f}s", flush=True)
    contact_sheet(images, labels).save(out_dir / "cfg_sweep.png")

    # 2) Step-count sweep — fixed CFG 7.5, DPM++.
    images, labels = [], []
    for steps in (3, 6, 12, 25):
        img, s = gen(pipe, num_inference_steps=steps, guidance_scale=7.5)
        images.append(img)
        labels.append(f"{steps} steps")
        print(f"{steps} steps: {s:.0f}s", flush=True)
    contact_sheet(images, labels).save(out_dir / "steps_sweep.png")

    # 3) Sampler comparison at a step count where solvers differ (8).
    images, labels = [], []
    for name, cls in (("Euler", EulerDiscreteScheduler),
                      ("DPM++ 2M", DPMSolverMultistepScheduler),
                      ("DDIM", DDIMScheduler)):
        pipe.scheduler = cls.from_config(pipe.scheduler.config)
        img, s = gen(pipe, num_inference_steps=8, guidance_scale=7.5)
        images.append(img)
        labels.append(name)
        print(f"{name} @ 8 steps: {s:.0f}s", flush=True)
    contact_sheet(images, labels).save(out_dir / "sampler_sweep.png")
    print(f"wrote 3 contact sheets to {out_dir}")


if __name__ == "__main__":
    main()
