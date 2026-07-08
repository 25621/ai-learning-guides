"""A vanilla VAE on MNIST — and the two things it buys you over a plain AE.

A VAE's encoder outputs a small Gaussian *cloud* per image (a mean and a
variance) instead of a single point, and the ELBO loss gently presses all those
clouds toward one standard normal. That regularization is what lets you throw
away the encoder, draw z ~ N(0, I), and decode a brand-new digit — something a
plain AE cannot do. The cost is blur: averaging over plausible reconstructions
plus the sampling noise softens the output.

This project trains a VAE and a same-sized AE side by side and shows both effects
directly: the VAE generates coherent digits from noise (the AE makes garbage),
but the VAE's reconstructions are blurrier than the AE's.

    python vae.py --data-dir data      # ~3 min on CPU (trains an AE and a VAE)

Reuses the shared conv models from project 06 via sys.path.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "06-tiny-ae-on-mnist"))
from vae_lib import AE, VAE, mnist_loader, infinite  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
ZDIM = 16


def train_vae(model, data_dir, steps, lr=1e-3, beta=1.0):
    loader = infinite(mnist_loader(data_dir, 128, True))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, _ = next(loader)
        loss, recon, kl = model.loss(x, beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 400 == 0:
            print(f"  [vae] step {step}/{steps} | recon {recon.item():.1f} | "
                  f"kl {kl.item():.1f} | {step/(time.time()-t0):.1f} it/s", flush=True)


def train_ae(model, data_dir, steps, lr=1e-3):
    loader = infinite(mnist_loader(data_dir, 128, True))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        x, _ = next(loader)
        loss = model.loss(x)
        opt.zero_grad(); loss.backward(); opt.step()


def grid(imgs, path, title, rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = imgs.size(0); cols = 10; rows = rows or (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows * 1.0), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for k, ax in enumerate(axes.flat if rows > 1 else axes):
        ax.axis("off")
        if k < n:
            ax.imshow(imgs[k, 0], cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title, fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_rows(rows, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = len(rows); C = rows[0].size(0)
    fig, axes = plt.subplots(R, C, figsize=(C, R * 1.0), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for r in range(R):
        for c in range(C):
            axes[r][c].imshow(rows[r][c, 0], cmap="gray", vmin=0, vmax=1)
            axes[r][c].axis("off")
        fig.text(0.5, 0, "", ha="center")
    for r, lab in enumerate(labels):
        axes[r][0].set_ylabel(lab, fontsize=9)
        fig.text(0.075, 1 - (r + 0.5) / R * 0.86 - 0.02, lab, ha="right",
                 fontsize=10, color="#0b0b0b")
    fig.suptitle(title, fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0.08, 0, 1, 0.93])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--vae-steps", type=int, default=2500)
    ap.add_argument("--ae-steps", type=int, default=1500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    print("training VAE ...")
    vae = VAE(zdim=ZDIM); train_vae(vae, args.data_dir, args.vae_steps); vae.eval()
    print("training a same-size AE for comparison ...")
    ae = AE(zdim=ZDIM); train_ae(ae, args.data_dir, args.ae_steps); ae.eval()

    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": vae.state_dict(), "zdim": ZDIM}, CKPT / "vae.pt")

    test = mnist_loader(args.data_dir, 128, train=False)
    x, _ = next(iter(test)); x = x[:10]
    with torch.no_grad():
        vae_recon, mu, logvar = vae(x)
        ae_recon, _ = ae(x)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean().item()
        samples = vae.sample(30)
        ae_rand = ae.dec(torch.randn(10, ZDIM))
        vae_rand = vae.dec(torch.randn(10, ZDIM))

    # 1) VAE samples from the prior — it can generate.
    grid(samples, OUT / "samples.png",
         "VAE samples: draw z ~ N(0, I), decode → brand-new digits")

    # 2) Reconstruction sharpness: AE vs VAE (VAE is blurrier).
    plot_rows([x, ae_recon, vae_recon], ["input", "AE rebuild", "VAE rebuild"],
              OUT / "reconstructions.png",
              "Reconstructions: the VAE is blurrier than the AE (it averages + adds noise)")

    # 3) Generation: only the VAE's regularized latent makes random codes valid.
    plot_rows([ae_rand, vae_rand], ["AE", "VAE"],
              OUT / "generation.png",
              "Decoding random z ~ N(0, I): AE → garbage, VAE → digits")

    (OUT / "results.csv").write_text(
        f"metric,value\nlatent_dim,{ZDIM}\nvae_recon_bce,"
        f"{torch.nn.functional.binary_cross_entropy(vae_recon, x, reduction='none').sum((1,2,3)).mean():.1f}\n"
        f"vae_kl,{kl:.2f}\n")
    print(f"VAE KL to prior: {kl:.2f} nats/image")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
