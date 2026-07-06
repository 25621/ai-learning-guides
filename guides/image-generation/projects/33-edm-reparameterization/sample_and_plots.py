"""Sample the trained EDM model with 18-step Heun and build the README figures.

Run (after train_edm.py):
    python sample_and_plots.py
"""

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
import plot_style as ps  # noqa: E402

from edm import EDMDenoiser, heun_sample, karras_sigmas  # noqa: E402


def plot_schedule(out_path):
    fig, ax = ps.new_axes(6.4, 3.6)
    for n, marker in ((18, "o"), (50, ".")):
        ax.plot(range(n), karras_sigmas(n), color=ps.SERIES[0] if n == 18 else ps.SERIES[1],
                linewidth=2, marker=marker, markersize=5 if n == 18 else 3)
    ax.annotate("18 steps", (5, karras_sigmas(18)[5] * 1.4), color=ps.INK_SECONDARY,
                fontsize=10, fontweight="bold")
    ax.annotate("50 steps", (30, karras_sigmas(50)[30] * 2.5), color=ps.INK_SECONDARY,
                fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ps.finish(fig, ax, "Karras sigma schedule (rho = 7): dense where it matters",
              "step index", "sigma", out_path)


def plot_loss_by_sigma(csv_path, out_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    sigmas = torch.tensor([float(r["sigma"]) for r in rows])
    losses = torch.tensor([float(r["unweighted_mse"]) for r in rows])
    # Bin by log-sigma and take the median raw MSE per bin.
    edges = torch.logspace(-2.2, 1.2, 18)
    centers, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (sigmas >= lo) & (sigmas < hi)
        if mask.sum() >= 10:
            centers.append(float((lo * hi).sqrt()))
            medians.append(float(losses[mask].median()))
    fig, ax = ps.new_axes(6.4, 3.6)
    ax.plot(centers, medians, color=ps.SERIES[0], linewidth=2, marker="o", markersize=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ps.finish(fig, ax, "Raw denoising MSE by sigma (before the EDM weight)",
              "sigma (log)", "median ||D - x0||^2", out_path)


def main():
    device = "cpu"
    ckpt = torch.load(HERE / "checkpoints/edm_mnist.pt", map_location=device,
                      weights_only=True)
    model = EDMDenoiser().to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    torch.manual_seed(7)
    x0 = heun_sample(model, (64, 1, 28, 28), n_steps=18, device=device)
    save_image(F.interpolate(x0.clamp(-1, 1), scale_factor=2, mode="nearest"),
               out_dir / "samples_heun18.png", nrow=8, normalize=True,
               value_range=(-1, 1))
    print("wrote", out_dir / "samples_heun18.png")

    plot_schedule(out_dir / "karras_schedule.png")
    plot_loss_by_sigma(HERE / "outputs/loss_by_sigma.csv",
                       out_dir / "loss_by_sigma.png")


if __name__ == "__main__":
    main()
