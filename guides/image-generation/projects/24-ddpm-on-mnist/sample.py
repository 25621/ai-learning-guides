"""Sample from a trained MNIST DDPM with the full T-step ancestral loop.

Run:
    python sample.py --ckpt checkpoints/mnist_linear.pt

Writes an 8x8 sample grid and a denoising-trajectory strip (the same batch
photographed at decreasing noise levels) to outputs/.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from diffusion import GaussianDiffusion
from unet import UNet


def load_ema_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = UNet().to(device)
    model.load_state_dict(ckpt["ema"])  # sample from the EMA weights
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/mnist_linear.pt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, ckpt = load_ema_model(args.ckpt, args.device)
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule=ckpt["schedule"], device=args.device)

    # Photograph the batch at these fractions of the schedule (1.0 = pure noise).
    fracs = (0.999, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05)
    snapshot_ts = tuple(int(f * (ckpt["T"] - 1)) for f in fracs)
    x0, snaps = diffusion.p_sample_loop(
        model, (args.n, 1, 28, 28), device=args.device, snapshot_ts=snapshot_ts
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    up = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")  # noqa: E731
    save_image(
        up(x0), out_dir / "samples.png", nrow=8, normalize=True, value_range=(-1, 1)
    )

    # Trajectory strip: the first 8 samples of the batch, one row per noise
    # level, from pure noise (top) to the final image (bottom).
    rows = [snaps[t][:8] for t in sorted(snaps, reverse=True)] + [x0[:8]]
    save_image(
        up(torch.cat(rows)), out_dir / "trajectory.png", nrow=8, normalize=True,
        value_range=(-1, 1),
    )
    print(f"wrote {out_dir}/samples.png and {out_dir}/trajectory.png")


if __name__ == "__main__":
    main()
