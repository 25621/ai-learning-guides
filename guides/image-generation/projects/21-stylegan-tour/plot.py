"""Build the report figures from stylegan_tour.py's saved arrays."""

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


def grid(images, ncols, path, titles=None, suptitle=None, figsize_per=1.6):
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_per, nrows * figsize_per), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            if titles is not None:
                ax.set_title(titles[i], fontsize=8, color=ps.INK)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, color=ps.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94] if suptitle else None)
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    samples = np.load(OUT / "samples.npy")
    grid(list(samples), 4, OUT / "samples.png",
         suptitle="Unconditional samples (truncation ψ=0.7)")

    trunc = np.load(OUT / "truncation.npy")
    psis = np.load(OUT / "truncation_psis.npy")
    grid(list(trunc), len(trunc), OUT / "truncation_sweep.png",
         titles=[f"ψ={p:.2f}" for p in psis],
         suptitle="Truncation trick: ψ=0 (mean face) → ψ=1 (full diversity, more artifacts)",
         figsize_per=2.3)

    mixed = np.load(OUT / "style_mixing.npy")
    crossovers = np.load(OUT / "style_mixing_crossovers.npy")
    titles = []
    for i, c in enumerate(crossovers):
        if i == 0:
            titles.append("A (source)")
        elif i == len(crossovers) - 1:
            titles.append("B (source)")
        else:
            titles.append(f"A[0:{c}) + B[{c}:18)")
    grid(list(mixed), len(mixed), OUT / "style_mixing.png", titles=titles,
         suptitle="Style mixing: crossover layer controls how much of A's identity survives",
         figsize_per=2.1)


if __name__ == "__main__":
    main()
