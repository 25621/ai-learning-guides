"""Bits-per-dimension: the 'no model' floor, and how easily structure beats it.

Bits-per-dim (bpd) is the standard likelihood metric for images: the average
number of bits a model needs to encode one pixel-channel value. Lower is better.
The dumbest possible model calls every value 0-255 equally likely; it needs
exactly log2(256) = 8 bits per dimension no matter what the data is. That 8.0 is
the ceiling every real generative model must beat.

This script computes that floor and then two trivial 'models' that already beat
it by exploiting obvious structure, on MNIST and CIFAR-10:

  - global histogram: one 256-bin distribution over ALL pixel values (learns
    e.g. that MNIST is mostly black);
  - per-pixel histogram: a separate 256-bin distribution for each pixel position
    (learns that the corners are ALWAYS black — zero bits there).

bpd of an independent model is just the average entropy of its per-value
distribution, so no training is needed — only counting.

    python bpd_baseline.py --data-dir data      # ~15s on CPU, no training
"""

import argparse
from pathlib import Path

import numpy as np
from torchvision import datasets

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
EPS = 1e-12


def load_mnist(data_dir, n=10000):
    ds = datasets.MNIST(str(data_dir), train=True, download=True)
    x = ds.data.numpy()[:n].astype(np.int64)          # (n,28,28) uint8 0-255
    return x.reshape(n, -1)                            # (n, 784)


def load_cifar(data_dir, n=10000):
    arr = np.load(Path(data_dir) / "cifar10_train.npz")
    x = arr["images"][:n].astype(np.int64)            # (n,32,32,3)
    return x.reshape(len(x), -1)                       # (n, 3072)


def uniform_bpd():
    """Every value in 0..255 equally likely -> exactly 8 bits per dim."""
    return 8.0


def global_hist_bpd(X):
    """One shared 256-bin distribution over all pixel values. bpd = entropy of
    that distribution in bits."""
    counts = np.bincount(X.reshape(-1), minlength=256).astype(np.float64)
    p = counts / counts.sum()
    return float(-(p * np.log2(p + EPS)).sum())


def per_pixel_entropy(X):
    """Per-position 256-bin distributions. Returns (mean_bpd, per_position_bits).
    Each position's cost is its own entropy; the mean over positions is the bpd
    of the independent per-pixel model."""
    n, d = X.shape
    bits = np.empty(d)
    for j in range(d):
        counts = np.bincount(X[:, j], minlength=256).astype(np.float64)
        p = counts / counts.sum()
        bits[j] = -(p * np.log2(p + EPS)).sum()
    return float(bits.mean()), bits


def plot_bars(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    models = ["uniform\n(no model)", "global\nhistogram", "per-pixel\nhistogram"]
    fig, ax = ps.new_axes(7.0, 4.2)
    x = np.arange(len(models)); w = 0.38
    ax.bar(x - w / 2, results["mnist"], w, color=ps.SERIES[0], label="MNIST")
    ax.bar(x + w / 2, results["cifar"], w, color=ps.SERIES[1], label="CIFAR-10")
    ax.axhline(8.0, color=ps.SERIES[2], linewidth=1.2, linestyle="--",
               label="8.0 bpd floor")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0, 9)
    for xi, (m, c) in enumerate(zip(results["mnist"], results["cifar"])):
        ax.text(xi - w / 2, m + 0.1, f"{m:.2f}", ha="center", fontsize=8, color=ps.INK_SECONDARY)
        ax.text(xi + w / 2, c + 0.1, f"{c:.2f}", ha="center", fontsize=8, color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Bits per dimension: trivial models already beat the 8.0 floor",
              "", "bits per dimension (lower is better)", path)


def plot_entropy_map(bits_mnist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = bits_mnist.reshape(28, 28)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    im = ax.imshow(grid, cmap="magma", vmin=0)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("bits", fontsize=9)
    ax.set_title("Per-pixel cost on MNIST:\ncorners are free, strokes are expensive",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    mnist = load_mnist(args.data_dir, args.n)
    cifar = load_cifar(args.data_dir, args.n)

    floor = uniform_bpd()
    m_global = global_hist_bpd(mnist)
    c_global = global_hist_bpd(cifar)
    m_pp, m_bits = per_pixel_entropy(mnist)
    c_pp, _ = per_pixel_entropy(cifar)

    results = {"mnist": [floor, m_global, m_pp], "cifar": [floor, c_global, c_pp]}
    plot_bars(results, OUT / "bpd_bars.png")
    plot_entropy_map(m_bits, OUT / "entropy_map.png")

    lines = ["model,mnist_bpd,cifar_bpd",
             f"uniform (no model),{floor:.3f},{floor:.3f}",
             f"global histogram,{m_global:.3f},{c_global:.3f}",
             f"per-pixel histogram,{m_pp:.3f},{c_pp:.3f}"]
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    print(f"{'model':<22}{'MNIST':>8}{'CIFAR':>8}")
    print(f"{'uniform (floor)':<22}{floor:>8.3f}{floor:>8.3f}")
    print(f"{'global histogram':<22}{m_global:>8.3f}{c_global:>8.3f}")
    print(f"{'per-pixel histogram':<22}{m_pp:>8.3f}{c_pp:>8.3f}")
    print(f"\nfree pixels on MNIST (0 bits): {(m_bits < 0.01).sum()} / {mnist.shape[1]}")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
