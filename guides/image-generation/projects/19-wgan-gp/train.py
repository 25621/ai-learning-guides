"""WGAN-GP on MNIST, contrasted with the older weight-clipping WGAN.

Both configs use the Wasserstein critic loss (no sigmoid, no BCE) and the
same 5-critic-steps-per-generator-step schedule. They differ only in how the
critic is kept 1-Lipschitz:

`--config clip` clamps every critic weight to [-0.01, 0.01] after each step
                (the original WGAN). Simple, but it pushes weights to the
                clip boundary and starves the critic's capacity.
`--config gp`   penalizes ||grad_x D(x)||_2 away from 1 at points interpolated
                between real and fake (WGAN-GP). No clipping, and the critic
                actually has to look like a 1-Lipschitz function rather than
                being forced into a corner of weight space.

    python train.py --config clip --data-dir data   # ~3 min on CPU
    python train.py --config gp   --data-dir data   # ~3.5 min on CPU
    python train.py --plot
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

sys.path.insert(0, str(HERE.parent / "18-vanilla-gan-on-mnist"))
from dcgan import Discriminator, Generator, infinite, mnist_loader, weights_init  # noqa: E402

N_CRITIC = 5
LAMBDA_GP = 10.0
CLIP = 0.01
NZ = 100
BATCH = 64
STEPS = 300  # generator updates; each does N_CRITIC critic updates first


def gradient_penalty(D, x_real, x_fake):
    eps = torch.rand(x_real.size(0), 1, 1, 1)
    x_hat = (eps * x_real + (1 - eps) * x_fake).requires_grad_(True)
    d_hat = D(x_hat)
    grad = torch.autograd.grad(d_hat.sum(), x_hat, create_graph=True)[0]
    grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean(), grad_norm


def run(cfg_name, data_dir, seed=0):
    torch.manual_seed(seed)
    loader = infinite(mnist_loader(data_dir, batch_size=BATCH))

    G = Generator(nz=NZ, nc=1); G.apply(weights_init)
    D = Discriminator(nc=1, norm="none"); D.apply(weights_init)
    opt_g = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    history = []  # step, wasserstein_estimate, mean_grad_norm (gp only)
    t0 = time.time()
    for step in range(STEPS):
        grad_norms = []
        for _ in range(N_CRITIC):
            x_real, _ = next(loader)
            z = torch.randn(BATCH, NZ)
            x_fake = G(z).detach()

            if cfg_name == "gp":
                gp, gnorm = gradient_penalty(D, x_real, x_fake)
                grad_norms.append(gnorm.mean().item())
                loss_d = D(x_fake).mean() - D(x_real).mean() + LAMBDA_GP * gp
            else:
                loss_d = D(x_fake).mean() - D(x_real).mean()

            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            if cfg_name == "clip":
                with torch.no_grad():
                    for p in D.parameters():
                        p.clamp_(-CLIP, CLIP)

        z = torch.randn(BATCH, NZ)
        x_fake = G(z)
        loss_g = -D(x_fake).mean()
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        with torch.no_grad():
            w_est = (D(x_real).mean() - D(G(z)).mean()).item()
        history.append((step, w_est, np.mean(grad_norms) if grad_norms else float("nan")))

        if step % 50 == 0:
            print(f"[{cfg_name}] step {step} W_est={w_est:.3f} ({time.time() - t0:.0f}s)")

    G.eval()
    with torch.no_grad():
        z = torch.randn(64, NZ)
        samples = G(z).clamp(-1, 1)

    OUT.mkdir(exist_ok=True)
    np.savez(OUT / f"run_{cfg_name}.npz",
              history=np.array(history), samples=samples.numpy(), wall_time=time.time() - t0)
    print(f"[{cfg_name}] done in {time.time() - t0:.0f}s")


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    runs = {name: np.load(OUT / f"run_{name}.npz") for name in ("clip", "gp")}

    fig, ax = ps.new_axes(7.4, 4.2)
    for name, color in zip(("clip", "gp"), (ps.SERIES[2], ps.SERIES[0])):
        h = runs[name]["history"]
        ax.plot(h[:, 0], h[:, 1], color=color, linewidth=1.3, label=name)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax,
              "Wasserstein estimate D(real) - D(fake): gradient penalty tracks progress smoothly",
              "generator step", "Wasserstein estimate", OUT / "wasserstein_estimate.png")

    # Critic gradient-norm distribution — the Lipschitz constraint in action (gp only has this).
    fig, ax = ps.new_axes(7.0, 4.0)
    gp_hist = runs["gp"]["history"]
    ax.plot(gp_hist[:, 0], gp_hist[:, 2], color=ps.SERIES[0], linewidth=1.2)
    ax.axhline(1.0, color=ps.SERIES[1], linestyle="--", linewidth=1.2, label="target: ||grad|| = 1")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Gradient penalty keeps the critic's input-gradient norm near 1 (1-Lipschitz)",
              "generator step", "mean ||grad_x D(x_hat)||", OUT / "grad_norm.png")

    fig, axes = plt.subplots(2, 8, figsize=(9, 2.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for r, name in enumerate(("clip", "gp")):
        imgs = runs[name]["samples"][:8]
        for c in range(8):
            img = (imgs[c, 0] + 1) / 2
            axes[r][c].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[r][c].axis("off")
        axes[r][0].axis("on")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for s in axes[r][0].spines.values():
            s.set_visible(False)
        axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9, color=ps.INK)
    fig.suptitle("Weight clipping (top) vs gradient penalty (bottom), same step budget", fontsize=11, color=ps.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "samples.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'samples.png'}")

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "wall_time_s", "final_wasserstein_estimate", "mean_grad_norm_last_50"])
        for name in ("clip", "gp"):
            h = runs[name]["history"]
            gnorm = h[-50:, 2]
            gnorm_mean = float(np.nanmean(gnorm)) if not np.all(np.isnan(gnorm)) else float("nan")
            w.writerow([name, f"{float(runs[name]['wall_time']):.1f}", f"{h[-1, 1]:.3f}",
                        f"{gnorm_mean:.3f}" if gnorm_mean == gnorm_mean else "n/a"])
    print(f"wrote {OUT / 'results.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["clip", "gp"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    if args.plot:
        make_plots()
    else:
        run(args.config, args.data_dir)
