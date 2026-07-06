"""Fréchet Inception Distance (FID), implemented from scratch.

FID scores how close a set of generated images is to a set of real ones. The
recipe, in full:

  1. push both sets through a pretrained InceptionV3 and take the 2048-d pooled
     feature of each image;
  2. summarize each set as a Gaussian — a mean vector and a covariance matrix;
  3. measure the Fréchet (Wasserstein-2) distance between the two Gaussians:

        FID = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2 (C_r C_g)^{1/2})

The only subtlety is the matrix square root; we compute it from a symmetric
eigendecomposition (no scipy). Lower FID = the feature clouds overlap more = the
fakes look statistically like the reals.

To *prove* the metric behaves, we don't need a generator — we corrupt real
images by known amounts and watch FID climb monotonically, and we check that FID
between two disjoint real sets is small (and FID of a set with itself is 0).

    python fid.py --data-dir data      # ~4 min on CPU (Inception features dominate)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def load_cifar(data_dir, n, seed=0, split="test"):
    arr = np.load(Path(data_dir) / f"cifar10_{split}.npz")
    imgs = torch.from_numpy(arr["images"]).float().permute(0, 3, 1, 2) / 255.0  # (N,3,32,32) [0,1]
    imgs = imgs * 2 - 1                                                          # [-1,1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(imgs))[:n]
    return imgs[idx]


def build_extractor():
    net = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
    net.eval()
    return create_feature_extractor(net, {"avgpool": "feat"})


@torch.no_grad()
def inception_features(images, extractor, bs=64):
    """images in [-1,1], (N,3,H,W) -> (N,2048)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), bs):
        x = (images[i:i + bs] + 1) / 2                          # -> [0,1]
        x = F.interpolate(x, size=299, mode="bilinear", align_corners=False)
        x = (x - mean) / std
        out.append(extractor(x)["feat"].flatten(1))
    return torch.cat(out)


# ---- the from-scratch FID math ------------------------------------------- #
def gaussian_stats(feats):
    mu = feats.mean(0)
    c = feats - mu
    cov = c.T @ c / (feats.size(0) - 1)
    return mu, cov


def sqrtm_psd(mat):
    """Square root of a (near-)PSD symmetric matrix via eigendecomposition."""
    vals, vecs = torch.linalg.eigh((mat + mat.T) / 2)
    return vecs @ torch.diag(vals.clamp(min=0).sqrt()) @ vecs.T


def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    # A tiny ridge keeps the covariances positive-definite: with fewer samples
    # than feature dimensions (512 < 2048) the raw covariance is singular, and
    # the matrix-sqrt of a singular matrix picks up numerical error that can even
    # push an identical-set FID slightly below zero.
    d = cov1.size(0)
    ridge = eps * torch.eye(d)
    cov1 = cov1 + ridge
    cov2 = cov2 + ridge
    s1 = sqrtm_psd(cov1)
    cross = sqrtm_psd(s1 @ cov2 @ s1).trace()
    val = (mu1 - mu2).square().sum() + cov1.trace() + cov2.trace() - 2 * cross
    return float(val.clamp(min=0))


def fid(feats_a, feats_b):
    return frechet_distance(*gaussian_stats(feats_a), *gaussian_stats(feats_b))


# ---- corruptions --------------------------------------------------------- #
def add_noise(x, sigma):
    return (x + sigma * torch.randn_like(x)).clamp(-1, 1)


def blur(x, k):
    if k <= 1:
        return x
    return F.avg_pool2d(x, k, 1, k // 2)


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()


def plot_curves(noise_x, noise_fid, blur_x, blur_fid, baseline, path):
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    fig, ax = ps.new_axes(7.0, 4.4)
    ax.plot(noise_x, noise_fid, "-o", color=ps.SERIES[2], label="add Gaussian noise")
    ax.plot(blur_x, blur_fid, "-o", color=ps.SERIES[0], label="blur")
    ax.axhline(baseline, color=ps.SERIES[1], linestyle="--", linewidth=1.2,
               label=f"real vs real ({baseline:.1f})")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "FID rises monotonically as images are corrupted",
              "corruption strength (noise std, or blur kernel/10)", "FID (lower is better)", path)


def plot_corrupted(clean, sigmas, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = len(sigmas)
    fig, axes = plt.subplots(rows, 8, figsize=(8, rows * 1.05), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for r, s in enumerate(sigmas):
        imgs = to_uint8(add_noise(clean[:8], s))
        for j in range(8):
            axes[r][j].imshow(imgs[j]); axes[r][j].axis("off")
        axes[r][0].set_ylabel(f"σ={s}", fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle("The same images at rising noise levels (what higher FID means)",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--n", type=int, default=512)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    print("loading InceptionV3 ...")
    ext = build_extractor()

    # Two disjoint real sets (A = reference, B = the set we corrupt).
    A = load_cifar(args.data_dir, args.n, seed=0, split="test")
    B = load_cifar(args.data_dir, args.n, seed=1, split="train")
    print(f"extracting features for {args.n}+{args.n} real images ...")
    fa = inception_features(A, ext)
    fb = inception_features(B, ext)

    self_fid = fid(fa, fa)                       # must be ~0
    baseline = fid(fa, fb)                       # real vs real, small
    print(f"FID(A, A)  = {self_fid:.3f}   (sanity: identical sets -> 0)")
    print(f"FID(A, B)  = {baseline:.3f}   (two disjoint real sets)")

    sigmas = [0.1, 0.2, 0.4, 0.8]
    noise_fid = []
    for s in sigmas:
        f = inception_features(add_noise(B, s), ext)
        noise_fid.append(fid(fa, f))
        print(f"FID(A, B+noise σ={s}) = {noise_fid[-1]:.1f}")

    kernels = [3, 5, 7]
    blur_fid = []
    for k in kernels:
        f = inception_features(blur(B, k), ext)
        blur_fid.append(fid(fa, f))
        print(f"FID(A, blur k={k})     = {blur_fid[-1]:.1f}")

    plot_curves(sigmas, noise_fid, [k / 10 for k in kernels], blur_fid, baseline,
                OUT / "fid_vs_corruption.png")
    plot_corrupted(A, [0.0, 0.2, 0.4, 0.8], OUT / "corruptions.png")

    lines = ["setting,fid",
             f"self (A vs A),{self_fid:.3f}",
             f"real vs real (A vs B),{baseline:.3f}"]
    for s, v in zip(sigmas, noise_fid):
        lines.append(f"noise sigma {s},{v:.2f}")
    for k, v in zip(kernels, blur_fid):
        lines.append(f"blur k {k},{v:.2f}")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
