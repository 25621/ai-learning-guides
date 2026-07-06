"""Invisible watermarking of a generator's outputs, with a detector and a
robustness study.

As AI images flood the internet, providers stamp an invisible, machine-readable
signal into every image they produce so it can later be proven synthetic. We
build the classic, reliable version: a **spread-spectrum watermark in the DCT
(frequency) domain**. A secret pseudo-random sign pattern is added to a band of
mid-frequency coefficients — low enough to survive blur and JPEG, high enough to
stay invisible. Detection correlates the (possibly attacked) image's
coefficients with the secret pattern: watermarked images light up, real images
score ~0.

The images we mark come from the phase-5 DDPM generator, but the watermark is
post-hoc and model-agnostic — exactly the "stamp it into the pixels afterward"
recipe. (SynthID-style *in-sampling* watermarks are the alternative; noted in
the README.)

    python watermark.py --data-dir data      # ~4 min on CPU

Reuses the phase-5 U-Net / DDPM (project 24) as the image source.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))

from diffusion import GaussianDiffusion  # noqa: E402
from train import EMA, infinite  # noqa: E402
from unet import UNet  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
T = 200
SIZE = 28      # generation resolution
RES = 112      # watermarking resolution — the generated digit is upscaled to
               # this before marking. Larger, smoother images have far more
               # low-energy high-frequency DCT coefficients to hide a mark in;
               # this is exactly why watermarks are invisible on real photos and
               # nearly impossible on a 28x28 thumbnail.


# --------------------------------------------------------------------------- #
# Image source: a plain unconditional DDPM (so we mark 'AI images').
# --------------------------------------------------------------------------- #
def train_generator(data_dir, out, steps=1200, seed=0):
    if out.exists():
        m = UNet(); m.load_state_dict(torch.load(out)["ema"]); return m.eval()
    torch.manual_seed(seed)
    model = UNet(); ema = EMA(model, 0.995)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batches = infinite(mnist_loader(data_dir))
    t0 = time.time()
    for step in range(1, steps + 1):
        loss = diffusion.loss(model, next(batches)[0])
        opt.zero_grad(); loss.backward(); opt.step(); ema.update(model)
        if step % 300 == 0:
            print(f"  [gen] {step}/{steps} loss {loss.item():.4f} "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": ema.shadow.state_dict()}, out)
    return ema.shadow.eval()


def mnist_loader(data_dir, bs=64):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)


# --------------------------------------------------------------------------- #
# A small orthonormal 2D DCT built from a basis matrix (no scipy/cv2 needed).
# --------------------------------------------------------------------------- #
def dct_matrix(N=RES):
    n = torch.arange(N).float()
    k = torch.arange(N).float()[:, None]
    D = torch.cos(np.pi * (2 * n + 1) * k / (2 * N)) * np.sqrt(2.0 / N)
    D[0] *= 1 / np.sqrt(2)
    return D                                    # (N, N), orthonormal


DCT = dct_matrix(RES)


def dct2(x):      # x: (B,1,H,W)
    return DCT @ x @ DCT.t()


def idct2(X):
    return DCT.t() @ X @ DCT


# --------------------------------------------------------------------------- #
# The watermark: a secret sign pattern over a mid-frequency band.
# --------------------------------------------------------------------------- #
def make_key(rmin=20, rmax=52, seed=0):
    u = torch.arange(RES).float()
    r = torch.sqrt(u[:, None] ** 2 + u[None, :] ** 2)
    mask = (r >= rmin) & (r <= rmax)
    g = torch.Generator().manual_seed(seed)
    sign = torch.where(torch.rand(RES, RES, generator=g) > 0.5, 1.0, -1.0)
    return mask, sign


def embed(imgs, mask, sign, alpha):
    X = dct2(imgs)
    X = X + alpha * (mask * sign)[None, None]
    return idct2(X).clamp(-1, 1)


def detect_stat(imgs, mask, sign):
    """Normalised correlation of the image's mid-band DCT coefficients with the
    secret sign pattern. ~alpha for watermarked images, ~0 for clean ones."""
    X = dct2(imgs)
    coeff = X[:, 0][:, mask]                     # (B, n_band)
    w = sign[mask][None]                         # (1, n_band)
    return (coeff * w).mean(dim=1)               # (B,)


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return 10 * np.log10(4.0 / (mse + 1e-12))    # range is [-1,1] -> peak^2 = 4


# --------------------------------------------------------------------------- #
# Attacks for the robustness study.
# --------------------------------------------------------------------------- #
def attack(imgs, kind, amount):
    if kind == "noise":
        return (imgs + amount * torch.randn_like(imgs)).clamp(-1, 1)
    if kind == "blur":
        k = int(amount)
        return imgs if k <= 1 else F.avg_pool2d(imgs, k, 1, k // 2)
    if kind == "jpeg":  # coarse quantisation of the DCT coefficients
        X = dct2(imgs)
        q = amount
        return idct2(torch.round(X / q) * q).clamp(-1, 1)
    if kind == "crop":
        H = imgs.shape[-1]; c = int(H * amount); s = (H - c) // 2
        return F.interpolate(imgs[:, :, s:s + c, s:s + c], size=(H, H),
                             mode="bilinear", align_corners=False)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.08)
    ap.add_argument("--steps", type=int, default=1200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    print("training / loading the generator ...")
    model = train_generator(args.data_dir, CKPT / "gen.pt", args.steps)

    print("sampling images to watermark ...")
    with torch.no_grad():
        small, _ = diffusion.p_sample_loop(model, (args.n, 1, SIZE, SIZE))
    imgs = F.interpolate(small, size=(RES, RES), mode="bilinear", align_corners=False)

    mask, sign = make_key()
    print(f"watermark band: {int(mask.sum())} mid-frequency DCT coefficients")
    wm = embed(imgs, mask, sign, args.alpha)
    clean = imgs

    q_psnr = psnr(clean, wm)
    d_wm = detect_stat(wm, mask, sign).numpy()
    d_clean = detect_stat(clean, mask, sign).numpy()
    thresh = args.alpha / 2
    tpr = float((d_wm > thresh).mean())
    fpr = float((d_clean > thresh).mean())
    print(f"\ninvisibility: PSNR {q_psnr:.1f} dB (watermarked vs original)")
    print(f"detection: TPR={tpr:.0%}  FPR={fpr:.0%}  (threshold={thresh:.3f})")
    print(f"statistic  watermarked {d_wm.mean():.3f} +/- {d_wm.std():.3f} | "
          f"clean {d_clean.mean():.3f} +/- {d_clean.std():.3f}")

    sweeps = {
        "noise": [0.0, 0.1, 0.2, 0.3, 0.5],
        "blur": [1, 3, 5],
        "jpeg": [0.05, 0.1, 0.2, 0.4],
        "crop": [1.0, 0.9, 0.8],
    }
    robustness = {}
    for kind, amounts in sweeps.items():
        rates = []
        for a in amounts:
            d = detect_stat(attack(wm, kind, a), mask, sign).numpy()
            rates.append(float((d > thresh).mean()))
        robustness[kind] = (amounts, rates)
        print(f"robustness[{kind}]: " + ", ".join(f"{a}:{r:.0%}" for a, r in zip(amounts, rates)))

    plot_hist(d_wm, d_clean, thresh, OUT / "detection.png")
    plot_robustness(robustness, OUT / "robustness.png")
    plot_images(clean, wm, q_psnr, OUT / "samples.png")

    lines = ["metric,value", f"psnr_db,{q_psnr:.2f}", f"TPR,{tpr:.3f}", f"FPR,{fpr:.3f}",
             f"stat_watermarked,{d_wm.mean():.4f}", f"stat_clean,{d_clean.mean():.4f}"]
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")
    print(f"\nwrote figures + {OUT/'results.csv'}")


def plot_hist(d_wm, d_clean, thresh, path):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps
    fig, ax = ps.new_axes(6.8, 4.2)
    lo = min(d_clean.min(), d_wm.min()); hi = max(d_clean.max(), d_wm.max())
    bins = np.linspace(lo, hi, 40)
    ax.hist(d_clean, bins=bins, color=ps.SERIES[2], alpha=0.8, label="clean (no mark)")
    ax.hist(d_wm, bins=bins, color=ps.SERIES[1], alpha=0.8, label="watermarked")
    ax.axvline(thresh, color=ps.INK_SECONDARY, linestyle="--", linewidth=1, label="threshold")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Detection statistic cleanly separates watermarked from clean",
              "correlation with the secret key", "count", path)


def plot_robustness(robustness, path):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps
    fig, ax = ps.new_axes(7.2, 4.2)
    labels = {"noise": "gaussian noise (std)", "blur": "blur kernel",
              "jpeg": "DCT quantiser step", "crop": "keep fraction"}
    colors = [ps.SERIES[0], ps.SERIES[1], ps.SERIES[2], ps.INK_MUTED]
    for i, (kind, (amounts, rates)) in enumerate(robustness.items()):
        xs = np.arange(len(amounts))
        ax.plot(xs + i * 0.0, rates, "-o", color=colors[i], label=labels[kind])
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("attack strength (increasing →)")
    ax.set_xticks([])
    ax.legend(frameon=False, fontsize=8)
    ps.finish(fig, ax, "Robustness: the mark survives noise, blur and JPEG; global crop breaks it",
              "attack strength (increasing →)", "detection rate", path)


def plot_images(clean, wm, q_psnr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def row(imgs, n=8):
        r = torch.cat([imgs[i, 0] for i in range(n)], dim=1)
        return ((r.clamp(-1, 1) + 1) * 127.5).byte().numpy()

    diff = (wm - clean)[:8]
    dvis = (diff[:, 0] - diff.min()) / (diff.max() - diff.min() + 1e-8)
    drow = torch.cat([dvis[i] for i in range(8)], dim=1).numpy()
    fig, axes = plt.subplots(3, 1, figsize=(9, 3.4), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    axes[0].imshow(row(clean), cmap="gray", vmin=0, vmax=255)
    axes[0].set_ylabel("original", fontsize=9, rotation=0, ha="right", va="center")
    axes[1].imshow(row(wm), cmap="gray", vmin=0, vmax=255)
    axes[1].set_ylabel("watermarked", fontsize=9, rotation=0, ha="right", va="center")
    axes[2].imshow(drow, cmap="magma")
    axes[2].set_ylabel("difference\n(x amplified)", fontsize=8, rotation=0, ha="right", va="center")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"The watermark is invisible (PSNR {q_psnr:.0f} dB); "
                 "only the amplified difference reveals the ripple",
                 fontsize=10, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
