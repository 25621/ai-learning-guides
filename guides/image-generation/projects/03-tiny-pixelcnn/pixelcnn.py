"""A tiny PixelCNN on (binarized) MNIST: autoregressive image modeling.

PixelCNN factorizes an image as a product of per-pixel conditionals in raster
order:  p(x) = prod_i p(x_i | x_1..x_{i-1}).  A network predicts every pixel's
distribution at once, but *masked* convolutions guarantee each output only sees
pixels above-and-to-the-left — so the whole factorization trains in a single
forward/backward pass on the real image (teacher forcing).

We use the classic binarized-MNIST setup (each pixel is 0 or 1), which makes the
per-pixel distribution a single Bernoulli and the samples crisp. Two things this
project makes concrete:

  1. modeling pixel *dependencies* matters — an independent per-pixel model can
     match PixelCNN's marginal likelihood yet its samples are pure speckle,
     while PixelCNN's are coherent digits;
  2. sampling is painfully slow: 784 sequential forward passes per 28x28 image,
     because pixel i needs pixel i-1.

    python pixelcnn.py --data-dir data      # ~5 min on CPU (train + slow sampling)
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
LN2 = math.log(2)


class MaskedConv2d(nn.Conv2d):
    """Convolution whose kernel is zeroed for 'future' pixels. mask_type 'A'
    excludes the center pixel itself (used once, at the input); 'B' includes it."""

    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, kh, kw = self.weight.shape
        yc, xc = kh // 2, kw // 2
        self.mask[:, :, yc, xc + (mask_type == "B"):] = 0   # rest of center row
        self.mask[:, :, yc + 1:, :] = 0                     # all rows below

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN(nn.Module):
    def __init__(self, ch=64, n_layers=6):
        super().__init__()
        layers = [MaskedConv2d("A", 1, ch, 7, padding=3), nn.ReLU()]
        for _ in range(n_layers):
            layers += [MaskedConv2d("B", ch, ch, 3, padding=1),
                       nn.BatchNorm2d(ch), nn.ReLU()]
        layers += [nn.Conv2d(ch, 1, 1)]                     # one Bernoulli logit / pixel
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)                                  # (B,1,H,W) logits


def binarized_loader(data_dir, bs, train=True):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(str(data_dir), train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=2, drop_last=train)


def binarize(x01):
    return (x01 > 0.5).float()


def bpd_from_logits(logits, xb):
    """Bernoulli NLL in bits per dimension."""
    return F.binary_cross_entropy_with_logits(logits, xb) / LN2


# --------------------------------------------------------------------------- #
# Baselines computed directly from the data (no training).
# --------------------------------------------------------------------------- #
def independent_pixel_model(loader, n_batches=60):
    """Per-pixel Bernoulli p_i = P(pixel i = 1). Returns (bpd, p_map)."""
    s = torch.zeros(28 * 28); n = 0
    for k, (x, _) in enumerate(loader):
        xb = binarize(x).view(x.size(0), -1)
        s += xb.sum(0); n += x.size(0)
        if k + 1 >= n_batches:
            break
    p = (s / n).clamp(1e-4, 1 - 1e-4)
    ent = -(p * p.log2() + (1 - p) * (1 - p).log2())        # bits per pixel
    return float(ent.mean()), p


@torch.no_grad()
def eval_bpd(model, loader, n_batches=20):
    model.eval()
    tot, k = 0.0, 0
    for x, _ in loader:
        xb = binarize(x)
        tot += bpd_from_logits(model(xb * 2 - 1), xb).item()
        k += 1
        if k >= n_batches:
            break
    return tot / k


@torch.no_grad()
def sample_pixelcnn(model, n, progress_every=196):
    model.eval()
    x = torch.zeros(n, 1, 28, 28)
    t0 = time.time()
    for i in range(28):
        for j in range(28):
            p = torch.sigmoid(model(x * 2 - 1)[:, :, i, j])   # full forward per pixel
            x[:, :, i, j] = torch.bernoulli(p)
        if (i + 1) * 28 % progress_every == 0:
            print(f"    sampled {((i+1)*28)}/784 pixels ({time.time()-t0:.0f}s)", flush=True)
    return x, time.time() - t0


def sample_independent(p_map, n):
    return torch.bernoulli(p_map.view(1, 1, 28, 28).expand(n, 1, 28, 28))


def train(model, data_dir, steps, lr=1e-3):
    loader = binarized_loader(data_dir, 64, train=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    it = iter(loader); t0 = time.time()
    for step in range(1, steps + 1):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(loader); x, _ = next(it)
        xb = binarize(x)
        model.train()
        loss = bpd_from_logits(model(xb * 2 - 1), xb)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0:
            print(f"  step {step}/{steps} | train bpd {loss.item():.3f} | "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)


def grid(x, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = x.size(0); cols = 8; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k < n:
            ax.imshow(x[k, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title, fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_bpd(indep_bpd, pcnn_bpd, path):
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps
    names = ["uniform\n(floor)", "independent\nper-pixel", "PixelCNN\n(this)"]
    vals = [1.0, indep_bpd, pcnn_bpd]
    fig, ax = ps.new_axes(6.0, 4.2)
    colors = [ps.SERIES[2], ps.SERIES[1], ps.SERIES[0]]
    ax.bar(range(3), vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, color=ps.INK_SECONDARY)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(3)); ax.set_xticklabels(names)
    ps.finish(fig, ax, "Binarized-MNIST bits-per-dim (lower is better)",
              "", "bits per dimension", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--n-samples", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    # Independent per-pixel baseline (no training) + its samples.
    indep_bpd, p_map = independent_pixel_model(binarized_loader(args.data_dir, 256, True))
    print(f"independent per-pixel baseline: {indep_bpd:.3f} bpd (uniform floor 1.0)")

    model = PixelCNN()
    print(f"PixelCNN params: {sum(p.numel() for p in model.parameters()):,}\ntraining ...")
    train(model, args.data_dir, args.steps)
    test_bpd = eval_bpd(model, binarized_loader(args.data_dir, 64, train=False))
    print(f"\nPixelCNN test bpd: {test_bpd:.3f}")

    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, CKPT / "pixelcnn.pt")

    print(f"sampling {args.n_samples} images row-by-row (the slow part) ...")
    x_pcnn, secs = sample_pixelcnn(model, args.n_samples)
    x_indep = sample_independent(p_map, args.n_samples)
    print(f"generated {args.n_samples} PixelCNN images in {secs:.0f}s "
          f"({secs/args.n_samples:.1f}s/image, 784 forward passes each)")

    grid(x_pcnn, OUT / "samples_pixelcnn.png",
         "PixelCNN samples — coherent digits (one pixel at a time)")
    grid(x_indep, OUT / "samples_independent.png",
         "Independent per-pixel samples — same marginals, pure speckle")
    plot_bpd(indep_bpd, test_bpd, OUT / "bpd_comparison.png")

    (OUT / "results.csv").write_text(
        "model,bpd\n"
        f"uniform floor,1.000\n"
        f"independent per-pixel,{indep_bpd:.4f}\n"
        f"PixelCNN,{test_bpd:.4f}\n"
        f"sample_seconds_per_image,{secs/args.n_samples:.2f}\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
