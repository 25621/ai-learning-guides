"""Annealed Langevin sampling from the learned score, plus all README figures.

Langevin dynamics at a fixed noise level:
    x <- x + (alpha/2) * s(x, sigma) + sqrt(alpha) * z

Annealing runs this at a ladder of decreasing sigmas (Song & Ermon 2019):
start where the noised density is one broad blob (easy to explore), finish
where it is essentially the data distribution (sharp but multi-modal).

Run (after train_score.py):
    python sample_and_plot.py
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
import plot_style as ps  # noqa: E402

from score_net import ScoreNet  # noqa: E402
from toy_data import DATASETS  # noqa: E402


@torch.no_grad()
def annealed_langevin(model, n=2000, levels=10, steps_per_level=80,
                      eps=0.1, sigma_min=0.05, sigma_max=8.0, seed=7):
    """Returns final samples and one snapshot per noise level.

    The step size must scale with the noise level: alpha = eps * sigma^2
    keeps the Langevin chain stable at every rung of the ladder (the score's
    magnitude goes like 1/sigma^2, so this makes each move a fixed *fraction*
    of the local structure). Set alpha too large at high sigma and the chain
    explodes; too small at low sigma and it never mixes.
    """
    torch.manual_seed(seed)
    sigmas = torch.logspace(
        torch.log10(torch.tensor(sigma_max)),
        torch.log10(torch.tensor(sigma_min)), levels
    )
    x = sigma_max * torch.randn(n, 2)
    snapshots = []
    for sigma in sigmas:
        alpha = eps * sigma**2
        sig = torch.full((n,), float(sigma))
        for _ in range(steps_per_level):
            x = x + 0.5 * alpha * model(x, sig) + alpha.sqrt() * torch.randn_like(x)
        snapshots.append((float(sigma), x.clone()))
    return x, snapshots


def load(dataset: str) -> ScoreNet:
    ckpt = torch.load(HERE / f"checkpoints/{dataset}.pt", weights_only=True)
    model = ScoreNet()
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def plot_score_field(model, dataset, out_path):
    """The learned vector field at low noise, over the true data."""
    lim = 5.0
    grid = torch.linspace(-lim, lim, 24)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    pts = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    with torch.no_grad():
        s = model(pts, torch.full((pts.size(0),), 0.3))
    data = DATASETS[dataset](1200, generator=torch.Generator().manual_seed(0))

    fig, ax = ps.new_axes(6.0, 6.0)
    ax.scatter(data[:, 0], data[:, 1], s=4, color=ps.SERIES[0], alpha=0.35,
               linewidths=0)
    ax.quiver(pts[:, 0], pts[:, 1], s[:, 0], s[:, 1], color=ps.INK_MUTED,
              width=0.0035, alpha=0.85)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ps.finish(fig, ax, f"Learned score field at sigma = 0.3 — {dataset}",
              "", "", out_path)


def plot_annealing(snapshots, dataset, out_path):
    """Samples at five points along the sigma ladder."""
    import matplotlib.pyplot as plt

    picks = [0, 2, 5, 7, len(snapshots) - 1]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.0 * len(picks), 3.2), dpi=100)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, i in zip(axes, picks):
        sigma, x = snapshots[i]
        ax.set_facecolor(ps.SURFACE)
        ax.scatter(x[:, 0], x[:, 1], s=2, color=ps.SERIES[0], alpha=0.4, linewidths=0)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(ps.BASELINE)
        ax.set_title(f"sigma = {sigma:.2f}", color=ps.INK_SECONDARY, fontsize=10)
    fig.suptitle(f"Annealed Langevin sampling — {dataset}", color=ps.INK,
                 fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_overlay(samples, dataset, out_path):
    data = DATASETS[dataset](2000, generator=torch.Generator().manual_seed(1))
    fig, ax = ps.new_axes(6.0, 6.0)
    ax.scatter(data[:, 0], data[:, 1], s=5, color=ps.INK_MUTED, alpha=0.3,
               linewidths=0, label="true data")
    ax.scatter(samples[:, 0], samples[:, 1], s=5, color=ps.SERIES[0], alpha=0.55,
               linewidths=0, label="Langevin samples")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9, markerscale=3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ps.finish(fig, ax, f"Langevin samples vs true data — {dataset}", "", "", out_path)


if __name__ == "__main__":
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    for dataset in DATASETS:
        model = load(dataset)
        samples, snaps = annealed_langevin(model)
        plot_score_field(model, dataset, out_dir / f"score_field_{dataset}.png")
        plot_annealing(snaps, dataset, out_dir / f"annealing_{dataset}.png")
        plot_overlay(samples, dataset, out_dir / f"samples_{dataset}.png")
