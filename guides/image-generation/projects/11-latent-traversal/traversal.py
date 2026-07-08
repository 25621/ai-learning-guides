"""Latent traversal: turn one latent dial at a time and watch the output change.

Once a VAE is trained, individual directions in its latent space often line up
with human-meaningful factors of variation — with no labels ever provided. A
*traversal* freezes every latent number but one, sweeps that single dial, and
decodes at each step.

CelebA faces aren't available in this offline CPU environment, so we use a
synthetic sprites dataset (in the spirit of dSprites) where the TRUE factors are
known — a white square's x-position, y-position, and size. Known factors are
actually a bonus: we can *check* which latent dial learned which factor, turning
"the model discovered the knobs" from a claim into a measurement.

    python traversal.py      # ~4 min on CPU, synthetic data (no download)

Reuses the shared conv VAE from project 06 via sys.path.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "06-tiny-ae-on-mnist"))
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
from vae_lib import VAE  # noqa: E402

OUT = HERE / "outputs"
ZDIM = 8
FACTORS = ["x position", "y position", "size"]


def make_sprites(n, seed=0):
    """White squares on a 28x28 black canvas. Returns (images, factors) where
    factors[:, (0,1,2)] = (x, y, scale) each in [0,1]."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.15, 0.85, size=(n, 3))
    imgs = np.zeros((n, 1, 28, 28), dtype=np.float32)
    for i in range(n):
        x, y, s = f[i]
        side = int(6 + s * 12)                       # 6..18 px
        cx, cy = int(4 + x * 20), int(4 + y * 20)
        x0, x1 = max(0, cx - side // 2), min(28, cx + side // 2)
        y0, y1 = max(0, cy - side // 2), min(28, cy + side // 2)
        imgs[i, 0, y0:y1, x0:x1] = 1.0
    return torch.tensor(imgs), torch.tensor(f, dtype=torch.float32)


def train(model, data, steps, beta=4.0, bs=128, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(data), (bs,))
        loss, recon, kl = model.loss(data[idx], beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 400 == 0:
            print(f"  step {step}/{steps} | recon {recon.item():.1f} | kl {kl.item():.1f} | "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)


@torch.no_grad()
def traversal_grid(model, path, labels, span=3.0, steps=9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = torch.zeros(1, ZDIM)
    fig, axes = plt.subplots(ZDIM, steps, figsize=(steps, ZDIM), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    vals = torch.linspace(-span, span, steps)
    for d in range(ZDIM):
        for k, v in enumerate(vals):
            z = base.clone(); z[0, d] = v
            img = model.dec(z)[0, 0]
            axes[d][k].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[d][k].set_xticks([]); axes[d][k].set_yticks([])
        axes[d][0].set_ylabel(labels[d], rotation=0, labelpad=24, fontsize=9,
                              va="center", color="#0b0b0b")
    fig.suptitle("Latent traversal: each row sweeps one latent dial from −3 to +3",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


@torch.no_grad()
def correlation(model, data, factors, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mu, _ = model.encode(data[:2000])
    mu = mu.numpy(); fac = factors[:2000].numpy()
    # |Pearson correlation| between each latent dim and each true factor.
    C = np.zeros((ZDIM, len(FACTORS)))
    for d in range(ZDIM):
        for j in range(len(FACTORS)):
            C[d, j] = abs(np.corrcoef(mu[:, d], fac[:, j])[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 5.2), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    im = ax.imshow(C, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FACTORS))); ax.set_xticklabels(FACTORS, rotation=20, ha="right")
    ax.set_yticks(range(ZDIM)); ax.set_yticklabels([f"z{d}" for d in range(ZDIM)])
    for d in range(ZDIM):
        for j in range(len(FACTORS)):
            ax.text(j, d, f"{C[d, j]:.2f}", ha="center", va="center",
                    color="white" if C[d, j] < 0.6 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("|correlation|", fontsize=9)
    ax.set_title("Which latent dial learned which true factor?", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--beta", type=float, default=4.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    data, factors = make_sprites(20000)
    model = VAE(zdim=ZDIM)
    print(f"training beta-VAE (beta={args.beta}) on sprites ...")
    train(model, data, args.steps, beta=args.beta)
    model.eval()

    C = correlation(model, data, factors, OUT / "factor_correlation.png")
    # Label each latent row with the factor it aligned with (if any).
    labels = []
    for d in range(ZDIM):
        j = int(C[d].argmax())
        labels.append(f"z{d}\n({FACTORS[j]})" if C[d, j] > 0.45 else f"z{d}")
    traversal_grid(model, OUT / "traversal.png", labels)

    # Report the best-aligned latent for each true factor.
    lines = ["factor,best_latent,abs_correlation"]
    print("\nbest latent dial per true factor:")
    for j, name in enumerate(FACTORS):
        d = int(C[:, j].argmax())
        print(f"  {name:12s} -> z{d}  (|r|={C[d, j]:.2f})")
        lines.append(f"{name},z{d},{C[d, j]:.3f}")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
