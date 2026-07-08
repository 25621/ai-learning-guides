"""beta-VAE sweep: watch reconstruction, KL, and posterior collapse trade off.

The beta-VAE multiplies the VAE's KL term by a knob beta. This script trains the
same small VAE at several beta values and measures three things per run:

  - reconstruction error (how faithfully it rebuilds inputs);
  - KL to the prior (how tidy/organized the latent space is);
  - active units (how many latent dimensions still carry information) — the
    direct measure of posterior collapse.

Sweeping beta from 0 upward walks through three regimes: sharp-but-messy latents
(low beta), organized-but-blurry (moderate beta), and collapse (high beta), where
the decoder ignores the latent entirely and every latent dimension goes silent.

    python betavae.py --data-dir data      # ~7 min on CPU (one VAE per beta)

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
from vae_lib import VAE, mnist_loader, infinite  # noqa: E402

OUT = HERE / "outputs"
ZDIM = 16
BETAS = [0.0, 1.0, 4.0, 10.0, 25.0]


def train(model, data_dir, steps, beta, lr=1e-3):
    loader = infinite(mnist_loader(data_dir, 128, True))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, _ = next(loader)
        loss, recon, kl = model.loss(x, beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()
    return time.time() - t0


@torch.no_grad()
def evaluate(model, loader, n_batches=20):
    """Return (recon_bce, kl_total, active_units, per_dim_kl)."""
    recons, kls, mus, logvars = [], [], [], []
    for k, (x, _) in enumerate(loader):
        xhat, mu, logvar = model(x)
        recons.append(torch.nn.functional.binary_cross_entropy(
            xhat, x, reduction="none").sum((1, 2, 3)))
        # per-dimension KL, averaged over the batch
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())      # (B, z)
        kls.append(kld)
        mus.append(mu)
        if k + 1 >= n_batches:
            break
    recon = torch.cat(recons).mean().item()
    per_dim_kl = torch.cat(kls).mean(0)                            # (z,)
    kl_total = per_dim_kl.sum().item()
    # An 'active unit' is a latent dim whose encoded means still vary across data.
    var_mu = torch.cat(mus).var(0)
    active = int((var_mu > 0.01).sum())
    return recon, kl_total, active, per_dim_kl.numpy()


def grid_row(model, x, path):
    pass


def plot_tradeoff(betas, recons, kls, actives, path):
    import matplotlib
    matplotlib.use("Agg")
    import plot_style as ps
    fig, ax = ps.new_axes(7.2, 4.4)
    x = np.arange(len(betas))
    ax.plot(x, recons, "-o", color=ps.SERIES[0], label="reconstruction BCE (↓ better rebuild)")
    ax.set_ylabel("reconstruction BCE", color=ps.SERIES[0], fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in betas])
    ax2 = ax.twinx()
    ax2.plot(x, kls, "-o", color=ps.SERIES[2], label="KL to prior")
    ax2.plot(x, actives, "-s", color=ps.SERIES[1], label="active latent units")
    ax2.set_ylabel("KL (nats)  /  active units", color=ps.INK_SECONDARY, fontsize=10)
    ax2.spines["top"].set_visible(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc="center right")
    ps.finish(fig, ax, "beta sweep: reconstruction, KL, and posterior collapse",
              "beta (KL weight)", "reconstruction BCE", path)


def plot_samples_and_recon(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    betas = [r["beta"] for r in results]
    R = len(betas); C = 10
    fig, axes = plt.subplots(R, C, figsize=(C, R * 1.0), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for r, res in enumerate(results):
        for c in range(C):
            axes[r][c].imshow(res["samples"][c, 0], cmap="gray", vmin=0, vmax=1)
            axes[r][c].axis("off")
        fig.text(0.075, 1 - (r + 0.5) / R * 0.9 - 0.015, f"β={betas[r]:g}",
                 ha="right", fontsize=10, color="#0b0b0b")
    fig.suptitle("Samples z ~ N(0,I) per β: sharp/messy → organized/blurry → collapsed",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0.08, 0, 1, 0.94])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--steps", type=int, default=1400)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    test = mnist_loader(args.data_dir, 128, train=False)
    results = []
    for beta in BETAS:
        torch.manual_seed(0)
        model = VAE(zdim=ZDIM)
        secs = train(model, args.data_dir, args.steps, beta)
        model.eval()
        recon, kl, active, _ = evaluate(model, test)
        with torch.no_grad():
            samples = model.sample(10)
        results.append(dict(beta=beta, recon=recon, kl=kl, active=active, samples=samples))
        print(f"β={beta:<5g} recon {recon:6.1f} | KL {kl:6.2f} | active units "
              f"{active}/{ZDIM} | {secs:.0f}s", flush=True)

    betas = [r["beta"] for r in results]
    plot_tradeoff(betas, [r["recon"] for r in results], [r["kl"] for r in results],
                  [r["active"] for r in results], OUT / "tradeoff.png")
    plot_samples_and_recon(results, OUT / "samples_per_beta.png")

    lines = ["beta,recon_bce,kl,active_units"]
    for r in results:
        lines.append(f"{r['beta']:g},{r['recon']:.2f},{r['kl']:.3f},{r['active']}")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
