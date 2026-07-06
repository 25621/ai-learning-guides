"""See the image manifold: PCA of real CIFAR-10 images vs. pure random noise.

A 32x32x3 image is a point in 3,072-dimensional space. Real photos do not fill
that space — they lie on a thin, curved surface (the manifold), while random
pixel noise scatters everywhere. This script makes that visible three ways:

  1. project 1,000 real images and 1,000 uniform-noise images to 2D with PCA and
     scatter them — the reals form a tight structured blob, the noise a diffuse
     cloud many times larger;
  2. plot the cumulative explained variance of each set — real images concentrate
     almost all their variance in a handful of directions (low intrinsic
     dimension), random noise spreads it evenly across all 3,072;
  3. show a few example images of each so "real vs noise" is concrete.

    python pca_manifold.py --data-dir data      # ~10s on CPU, no training

Loads CIFAR-10 from a cifar10_test.npz (images uint8 NHWC) in the data dir.
"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def load_cifar(data_dir, n, seed=0):
    npz = Path(data_dir) / "cifar10_test.npz"
    arr = np.load(npz)
    imgs = arr["images"].astype(np.float32) / 255.0     # (N,32,32,3) in [0,1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(imgs))[:n]
    return imgs[idx]


def pca_fit(X, k=None):
    """Return (components, explained_variance_ratio, mean). X: (N, D)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    # economy SVD; singular values^2 / (N-1) are the eigenvalues (variances)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (X.shape[0] - 1)
    evr = var / var.sum()
    comps = Vt
    if k:
        comps = comps[:k]
    return comps, evr, mean


def project(X, comps, mean):
    return (X - mean) @ comps.T


def plot_scatter(real2d, rand2d, path):
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    fig, ax = ps.new_axes(6.6, 5.4)
    ax.scatter(rand2d[:, 0], rand2d[:, 1], s=10, color=ps.SERIES[2], alpha=0.5,
               edgecolor="none", label="uniform noise")
    ax.scatter(real2d[:, 0], real2d[:, 1], s=10, color=ps.SERIES[0], alpha=0.7,
               edgecolor="none", label="real CIFAR-10")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ps.finish(fig, ax, "Real images vs. noise in the top-2 PCA plane (shared axes)",
              "principal component 1", "principal component 2", path)


def plot_variance(evr_real, evr_rand, path):
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    fig, ax = ps.new_axes(6.8, 4.4)
    kmax = 200
    ax.plot(np.arange(1, kmax + 1), np.cumsum(evr_real[:kmax]),
            color=ps.SERIES[0], label="real CIFAR-10")
    ax.plot(np.arange(1, kmax + 1), np.cumsum(evr_rand[:kmax]),
            color=ps.SERIES[2], label="uniform noise")
    ax.axhline(0.9, color=ps.BASELINE, linewidth=1, linestyle="--")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ps.finish(fig, ax, "Cumulative variance explained: real images are low-dimensional",
              "number of principal components", "fraction of variance captured", path)


def pairwise_distances(X, m=400):
    X = X[:m]
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(m, 1)
    return d[iu]


def plot_distances(dr, dn, path):
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    fig, ax = ps.new_axes(6.8, 4.4)
    lo = min(dr.min(), dn.min()); hi = max(dr.max(), dn.max())
    bins = np.linspace(lo, hi, 60)
    ax.hist(dr, bins=bins, color=ps.SERIES[0], alpha=0.8, density=True, label="real CIFAR-10")
    ax.hist(dn, bins=bins, color=ps.SERIES[2], alpha=0.8, density=True, label="uniform noise")
    ax.legend(frameon=False, fontsize=10)
    ps.finish(fig, ax, "Pairwise distances: real images cluster; noise fills a uniform shell",
              "Euclidean distance between two images", "density", path)


def plot_examples(real, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(1)
    noise = rng.random((8, 32, 32, 3))
    fig, axes = plt.subplots(2, 8, figsize=(9, 2.5), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for j in range(8):
        axes[0][j].imshow(real[j]); axes[0][j].axis("off")
        axes[1][j].imshow(noise[j]); axes[1][j].axis("off")
    axes[0][0].set_ylabel("real", fontsize=10)
    fig.text(0.09, 0.72, "real", fontsize=11, color="#0b0b0b", ha="right")
    fig.text(0.09, 0.30, "noise", fontsize=11, color="#0b0b0b", ha="right")
    fig.suptitle("What the two point-clouds actually are", fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0.1, 0, 1, 0.93])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    real = load_cifar(args.data_dir, args.n)
    rng = np.random.default_rng(0)
    rand = rng.random((args.n, 32, 32, 3)).astype(np.float32)   # uniform [0,1]

    Xr = real.reshape(args.n, -1)
    Xn = rand.reshape(args.n, -1)

    # Shared PCA plane fit on the combined set, so both clouds use the same axes.
    comps, _, mean = pca_fit(np.concatenate([Xr, Xn], 0), k=2)
    real2d = project(Xr, comps, mean)
    rand2d = project(Xn, comps, mean)

    # Per-set explained variance to show intrinsic dimension.
    _, evr_real, _ = pca_fit(Xr)
    _, evr_rand, _ = pca_fit(Xn)

    # Quantify the manifold: how many PCs to reach 90% variance, and the
    # relative "size" of each cloud in the shared plane.
    k90_real = int(np.argmax(np.cumsum(evr_real) >= 0.9) + 1)
    k90_rand = int(np.argmax(np.cumsum(evr_rand) >= 0.9) + 1)
    spread_real = real2d.std(0).mean()
    spread_rand = rand2d.std(0).mean()

    dr = pairwise_distances(Xr)
    dn = pairwise_distances(Xn)

    plot_scatter(real2d, rand2d, OUT / "pca_scatter.png")
    plot_variance(evr_real, evr_rand, OUT / "explained_variance.png")
    plot_distances(dr, dn, OUT / "distances.png")
    plot_examples(real, OUT / "examples.png")

    lines = ["metric,real,noise",
             f"pcs_for_90pct_variance,{k90_real},{k90_rand}",
             f"top2_variance_fraction,{evr_real[:2].sum():.3f},{evr_rand[:2].sum():.3f}",
             f"mean_pairwise_distance,{dr.mean():.2f},{dn.mean():.2f}",
             f"std_pairwise_distance,{dr.std():.2f},{dn.std():.2f}"]
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    print(f"PCs to reach 90% variance:  real {k90_real}  vs  noise {k90_rand} (of 3072)")
    print(f"variance in top-2 PCs:       real {evr_real[:2].sum():.1%}  vs  noise {evr_rand[:2].sum():.1%}")
    print(f"pairwise dist mean+/-std:    real {dr.mean():.1f}+/-{dr.std():.1f}  vs  noise {dn.mean():.1f}+/-{dn.std():.1f}")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
