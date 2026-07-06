"""Compare DDIM at several step counts against the full 1000-step DDPM loop,
all starting from the SAME initial noise x_T.

Run (after training project 24):
    python compare_samplers.py --ckpt ../24-ddpm-on-mnist/checkpoints/mnist_linear.pt

Writes:
    outputs/ddpm_vs_ddim.png   one row per sampler, same x_T per column
    outputs/eta_study.png      DDIM-50 with eta = 0, 0.5, 1.0
    outputs/timing.csv         wall-clock seconds per sampler
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
from diffusion import GaussianDiffusion  # noqa: E402
from unet import UNet  # noqa: E402

from ddim import DDIMSampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE.parent / "24-ddpm-on-mnist/checkpoints/mnist_linear.pt"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    model = UNet().to(args.device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule=ckpt["schedule"], device=args.device)

    torch.manual_seed(args.seed)
    x_T = torch.randn(args.n, 1, 28, 28, device=args.device)  # shared starting noise

    rows, timings = [], [("sampler", "steps", "seconds")]

    t0 = time.time()
    x_ddpm, _ = diffusion.p_sample_loop(
        model, x_T.shape, device=args.device, x_init=x_T
    )
    timings.append(("DDPM", diffusion.T, f"{time.time() - t0:.1f}"))
    rows.append(x_ddpm)

    for steps in (100, 50, 20, 10):
        t0 = time.time()
        x = DDIMSampler(diffusion, num_steps=steps, eta=0.0).sample(
            model, x_T.shape, device=args.device, x_init=x_T
        )
        timings.append(("DDIM eta=0", steps, f"{time.time() - t0:.1f}"))
        rows.append(x)

    up = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")  # noqa: E731
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(
        up(torch.cat(rows)), out_dir / "ddpm_vs_ddim.png", nrow=args.n,
        normalize=True, value_range=(-1, 1),
    )

    # eta study at a fixed 50 steps: how much fresh noise re-enters the loop.
    eta_rows = []
    for eta in (0.0, 0.5, 1.0):
        x = DDIMSampler(diffusion, num_steps=50, eta=eta).sample(
            model, x_T.shape, device=args.device, x_init=x_T
        )
        eta_rows.append(x)
    save_image(
        up(torch.cat(eta_rows)), out_dir / "eta_study.png", nrow=args.n,
        normalize=True, value_range=(-1, 1),
    )

    with open(out_dir / "timing.csv", "w", newline="") as f:
        csv.writer(f).writerows(timings)
    for row in timings[1:]:
        print(f"{row[0]:>10} {row[1]:>5} steps  {row[2]:>7}s")
    print(f"wrote {out_dir}/ddpm_vs_ddim.png, {out_dir}/eta_study.png, {out_dir}/timing.csv")


if __name__ == "__main__":
    main()
