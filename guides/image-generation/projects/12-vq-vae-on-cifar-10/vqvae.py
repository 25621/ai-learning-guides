"""A VQ-VAE on CIFAR-10: compress each image to an 8x8 grid of discrete codes.

Encoder shrinks 32x32x3 -> 8x8xD, a vector quantizer snaps each of the 64
positions to the nearest of 256 learned codebook entries (the discrete tokens),
and a decoder rebuilds the image. The non-differentiable lookup is trained
through with the straight-through estimator.

Outputs: reconstructions, the codebook-usage histogram (with perplexity), and a
visualization of the 8x8 token grids.

    python vqvae.py --data-dir data      # ~6 min on CPU
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vq_lib import Encoder, Decoder, VectorQuantizerEMA, perplexity, cifar_loader, infinite, cifar_tensor

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
K, D = 256, 32


class VQVAE(torch.nn.Module):
    def __init__(self, K=K, D=D, ch=64):
        super().__init__()
        self.enc = Encoder(D=D, ch=ch)
        # EMA codebook updates + dead-code re-init (the standard VQ-VAE-2 recipe):
        # a plain loss-trained codebook collapses to a couple of entries on this
        # data — see project 13. The straight-through estimator still carries the
        # encoder's gradient; only the *codebook* is updated by moving averages.
        self.vq = VectorQuantizerEMA(K=K, D=D, reinit=True, dead_after=128)
        self.dec = Decoder(D=D, ch=ch)

    def forward(self, x):
        z_e = self.enc(x)
        z_q, vq_loss, idx = self.vq(z_e)
        return self.dec(z_q), vq_loss, idx


def train(model, data_dir, steps, lr=2e-4):
    loader = infinite(cifar_loader(data_dir, 128, True))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        (x,) = next(loader)
        xhat, vq_loss, idx = model(x)
        recon = F.mse_loss(xhat, x)
        loss = recon + vq_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0:
            pplx, _ = perplexity(idx, K)
            print(f"  step {step}/{steps} | recon {recon.item():.4f} | "
                  f"perplexity {pplx:.0f}/{K} | {step/(time.time()-t0):.1f} it/s", flush=True)


def to_img(x):
    return ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()


def plot_recon(model, x, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with torch.no_grad():
        xhat, _, _ = model(x)
    a, b = to_img(x), to_img(xhat)
    fig, axes = plt.subplots(2, 10, figsize=(10, 2.2), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for j in range(10):
        axes[0][j].imshow(a[j]); axes[0][j].axis("off")
        axes[1][j].imshow(b[j]); axes[1][j].axis("off")
    fig.text(0.085, 0.72, "input", ha="right", fontsize=10, color="#0b0b0b")
    fig.text(0.085, 0.30, "VQ-VAE", ha="right", fontsize=10, color="#0b0b0b")
    fig.suptitle("VQ-VAE reconstructions (each image = an 8×8 grid of codes)",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0.09, 0, 1, 0.93])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_usage(counts, pplx, path):
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps
    order = counts.argsort(descending=True).numpy()
    fig, ax = ps.new_axes(7.2, 4.0)
    ax.bar(range(len(counts)), counts[order].numpy() / counts.sum().item() * 100,
           color=ps.SERIES[0], width=1.0)
    ax.set_xlim(0, len(counts))
    ps.finish(fig, ax, f"Codebook usage across all {len(counts)} entries "
              f"(perplexity {pplx:.0f}/{len(counts)})",
              "codebook entry (sorted by frequency)", "% of tokens", path)


def plot_tokens(model, x, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with torch.no_grad():
        _, _, idx = model(x[:6])
    imgs = to_img(x[:6]); grids = idx.numpy()
    fig, axes = plt.subplots(2, 6, figsize=(9, 3.2), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for j in range(6):
        axes[0][j].imshow(imgs[j]); axes[0][j].axis("off")
        axes[1][j].imshow(grids[j], cmap="tab20"); axes[1][j].axis("off")
    fig.text(0.08, 0.70, "image", ha="right", fontsize=10, color="#0b0b0b")
    fig.text(0.08, 0.28, "8×8 codes", ha="right", fontsize=10, color="#0b0b0b")
    fig.suptitle("Each image becomes an 8×8 grid of code indices (colors = codes)",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0.09, 0, 1, 0.93])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    model = VQVAE()
    print(f"VQ-VAE params: {sum(p.numel() for p in model.parameters()):,} | "
          f"codebook {K}x{D}, latent 8x8")
    train(model, args.data_dir, args.steps)
    model.eval()
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "K": K, "D": D}, CKPT / "vqvae.pt")

    test = cifar_tensor(args.data_dir, train=False, n=2000)
    with torch.no_grad():
        xhat, _, idx = model(test)
        recon = F.mse_loss(xhat, test).item()
    pplx, counts = perplexity(idx, K)
    used = int((counts > 0).sum())
    print(f"\ntest recon MSE {recon:.4f} | perplexity {pplx:.0f}/{K} | codes used {used}/{K}")

    plot_recon(model, test[:10], OUT / "reconstructions.png")
    plot_usage(counts, pplx, OUT / "codebook_usage.png")
    plot_tokens(model, test, OUT / "token_grids.png")
    (OUT / "results.csv").write_text(
        f"metric,value\ncodebook_size,{K}\nlatent_grid,8x8\n"
        f"test_recon_mse,{recon:.4f}\nperplexity,{pplx:.1f}\ncodes_used,{used}\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
