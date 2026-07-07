"""Figures for the StyleGAN2 optimization-based inversion."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
import plot_style as ps  # noqa: E402


def main():
    d = np.load(OUT / "stylegan_invert.npz")

    fig, ax = ps.new_axes(7.0, 4.0)
    h = d["history"]
    ax.plot(h[:, 0], h[:, 1], color=ps.SERIES[0], linewidth=1.4)
    ps.finish(fig, ax, "W+ optimization: pixel MSE vs Adam step", "step", "MSE",
              OUT / "loss_curve.png")

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), dpi=120)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, key, title in zip(axes, ("target", "final"), ("target (a StyleGAN2 sample)", "recovered (W+ optimization)")):
        ax.imshow(d[key]); ax.axis("off"); ax.set_title(title, fontsize=10, color=ps.INK)
    fig.suptitle(f"PSNR {float(d['psnr']):.1f} dB after {len(d['history'])} Adam steps", fontsize=11, color=ps.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "target_vs_recon.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'target_vs_recon.png'}")

    snaps = d["snapshots"]
    n = len(snaps)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.7, 1.9), dpi=120)
    fig.patch.set_facecolor(ps.SURFACE)
    steps_logged = h[:, 0][::10].tolist() + [h[-1, 0]]
    steps_logged = sorted(set(int(s) for s in steps_logged))[:n]
    for i, ax in enumerate(axes):
        ax.imshow(snaps[i]); ax.axis("off")
        ax.set_title(f"step {steps_logged[i]}", fontsize=8, color=ps.INK)
    fig.suptitle("Optimization progress: mean face → target identity", fontsize=11, color=ps.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "progression.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'progression.png'}")


if __name__ == "__main__":
    main()
