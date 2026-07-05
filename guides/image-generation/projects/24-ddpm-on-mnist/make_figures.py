"""Build the explanatory figures for the README: the forward process strip
and the training loss curve.

Run after train.py:
    python make_figures.py --log outputs/train_log.csv
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import plot_style as ps
from diffusion import GaussianDiffusion


def forward_process_strip(data_dir: str, out_path: Path, T: int = 300):
    """One real digit pushed through q(x_t | x_0) at increasing t."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    x0 = ds[1][0].unsqueeze(0)  # one clean image, shape (1, 1, 28, 28)

    diffusion = GaussianDiffusion(T=T, schedule="linear")
    fracs = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.999]
    ts = [int(f * (T - 1)) for f in fracs]
    torch.manual_seed(0)
    noise = torch.randn_like(x0)
    frames = [
        diffusion.q_sample(x0, torch.tensor([t]), noise) if t > 0 else x0 for t in ts
    ]
    strip = F.interpolate(torch.cat(frames), scale_factor=3, mode="nearest")
    save_image(strip, out_path, nrow=len(ts), normalize=True, value_range=(-1, 1))
    print(f"wrote {out_path}  (t = {ts})")


def loss_curve(log_path: str, out_path: Path):
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]

    fig, ax = ps.new_axes()
    ax.plot(steps, losses, color=ps.SERIES[0], linewidth=2)
    ax.set_yscale("log")
    ps.finish(
        fig, ax,
        title="DDPM on MNIST — noise-prediction MSE (log scale)",
        xlabel="training step",
        ylabel="loss",
        out_path=out_path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--log", default="outputs/train_log.csv")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    forward_process_strip(args.data_dir, out_dir / "forward_process.png")
    loss_curve(args.log, out_dir / "loss_curve.png")


if __name__ == "__main__":
    main()
