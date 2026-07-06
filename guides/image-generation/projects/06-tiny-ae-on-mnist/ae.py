"""A tiny autoencoder on MNIST: compress 784 pixels to 32 numbers and back.

Trains a small conv autoencoder, then shows the three things that matter:
  1. reconstructions — the 32-number code keeps enough to rebuild the digit;
  2. latent interpolation — decoding the average of a '3' code and an '8' code
     gives a believable in-between digit, proving the code learned structure;
  3. the gap the VAE fixes — sampling a random latent and decoding gives
     garbage, because a plain AE never organized its latent space.

    python ae.py --data-dir data      # ~2 min on CPU
"""

import argparse
import time
from pathlib import Path

import torch

from vae_lib import AE, mnist_loader, infinite

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"


def train(model, data_dir, steps, lr=1e-3):
    loader = infinite(mnist_loader(data_dir, 128, True))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, _ = next(loader)
        loss = model.loss(x)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0:
            print(f"  step {step}/{steps} | recon {loss.item():.1f} | "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)


@torch.no_grad()
def get_digit(loader, cls):
    for x, y in loader:
        m = y == cls
        if m.any():
            return x[m][0:1]


def plot_recon(model, loader, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, _ = next(iter(loader))
    x = x[:10]
    with torch.no_grad():
        xhat, _ = model(x)
    fig, axes = plt.subplots(2, 10, figsize=(10, 2.2), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for j in range(10):
        axes[0][j].imshow(x[j, 0], cmap="gray", vmin=0, vmax=1); axes[0][j].axis("off")
        axes[1][j].imshow(xhat[j, 0], cmap="gray", vmin=0, vmax=1); axes[1][j].axis("off")
    axes[0][0].set_ylabel("input", fontsize=9)
    fig.text(0.085, 0.72, "input", ha="right", fontsize=10, color="#0b0b0b")
    fig.text(0.085, 0.30, "rebuilt", ha="right", fontsize=10, color="#0b0b0b")
    fig.suptitle("Autoencoder reconstructions (784 → 32 → 784)", fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0.09, 0, 1, 0.93])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_interp(model, data_dir, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loader = mnist_loader(data_dir, 256, train=False)
    pairs = [(3, 8), (1, 7), (4, 9), (0, 6)]
    steps = 9
    fig, axes = plt.subplots(len(pairs), steps, figsize=(steps, len(pairs)), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    with torch.no_grad():
        for r, (a, b) in enumerate(pairs):
            za = model.enc(get_digit(loader, a))
            zb = model.enc(get_digit(loader, b))
            for k in range(steps):
                t = k / (steps - 1)
                z = (1 - t) * za + t * zb
                img = model.dec(z)[0, 0]
                axes[r][k].imshow(img, cmap="gray", vmin=0, vmax=1); axes[r][k].axis("off")
    fig.suptitle("Latent interpolation: decoding the path from one digit's code to another's",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_random(model, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with torch.no_grad():
        z = torch.randn(10, model.enc.fc.out_features)
        imgs = model.dec(z)
    fig, axes = plt.subplots(1, 10, figsize=(10, 1.2), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for j in range(10):
        axes[j].imshow(imgs[j, 0], cmap="gray", vmin=0, vmax=1); axes[j].axis("off")
    fig.suptitle("Decoding random latents from a plain AE → garbage (why we need a VAE)",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--zdim", type=int, default=32)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    model = AE(zdim=args.zdim)
    print(f"AE params: {sum(p.numel() for p in model.parameters()):,} | latent dim {args.zdim}")
    train(model, args.data_dir, args.steps)
    model.eval()

    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "zdim": args.zdim}, CKPT / "ae.pt")

    test = mnist_loader(args.data_dir, 128, train=False)
    with torch.no_grad():
        x, _ = next(iter(test))
        recon = model.loss(x).item()
    print(f"test reconstruction BCE (per image): {recon:.1f}")

    plot_recon(model, test, OUT / "reconstructions.png")
    plot_interp(model, args.data_dir, OUT / "interpolation.png")
    plot_random(model, OUT / "random_latents.png")
    (OUT / "results.csv").write_text(f"metric,value\nlatent_dim,{args.zdim}\n"
                                     f"test_recon_bce,{recon:.2f}\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
