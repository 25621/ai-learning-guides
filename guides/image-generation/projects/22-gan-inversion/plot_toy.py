"""Figures for the toy-scale optimization-vs-encoder inversion comparison."""

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
    d = np.load(OUT / "toy_invert.npz")
    targets, opt_recons, enc_recons = d["targets"], d["opt_recons"], d["enc_recons"]
    n = len(targets)

    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 4.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    row_labels = ["target", "optimization\n(150 steps)", "encoder\n(1 forward)"]
    for r, (imgs, label) in enumerate(zip((targets, opt_recons, enc_recons), row_labels)):
        for c in range(n):
            img = (imgs[c] + 1) / 2
            axes[r][c].imshow(img, cmap="gray", vmin=0, vmax=1); axes[r][c].axis("off")
        axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for s in axes[r][0].spines.values():
            s.set_visible(False)
        axes[r][0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8, color=ps.INK)
    fig.suptitle("Same 5 real digits, inverted two ways", fontsize=12, color=ps.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "toy_comparison.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'toy_comparison.png'}")

    rows = d["rows"]  # target, opt_mse, opt_time, enc_mse, enc_time
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    fig.patch.set_facecolor(ps.SURFACE)
    x = np.arange(n)
    width = 0.35
    for ax in axes:
        ax.set_facecolor(ps.SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=8)
        ax.grid(True, color=ps.GRID, linewidth=0.7, axis="y")

    axes[0].bar(x - width / 2, rows[:, 1], width, color=ps.SERIES[0], label="optimization")
    axes[0].bar(x + width / 2, rows[:, 3], width, color=ps.SERIES[2], label="encoder")
    axes[0].set_title("Reconstruction MSE (lower is better)", loc="left", fontsize=10, color=ps.INK)
    axes[0].set_xlabel("target #", fontsize=9, color=ps.INK_SECONDARY)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].bar(x - width / 2, rows[:, 2] * 1000, width, color=ps.SERIES[0], label="optimization")
    axes[1].bar(x + width / 2, rows[:, 4] * 1000, width, color=ps.SERIES[2], label="encoder")
    axes[1].set_yscale("log")
    axes[1].set_title("Wall-clock per image, ms (log scale)", loc="left", fontsize=10, color=ps.INK)
    axes[1].set_xlabel("target #", fontsize=9, color=ps.INK_SECONDARY)
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Optimization: better fit, paid fresh every time. Encoder: instant, one-time training cost.",
                 fontsize=11, color=ps.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "toy_speed_quality.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'toy_speed_quality.png'}")


if __name__ == "__main__":
    main()
