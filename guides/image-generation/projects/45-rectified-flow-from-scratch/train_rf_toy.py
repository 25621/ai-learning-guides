"""Rectified flow on 2D toy data: train, sample, and DRAW the trajectories.

Run:
    python train_rf_toy.py            # ~1 min on CPU
"""

import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "30-score-matching-from-scratch"))
import plot_style as ps  # noqa: E402
from toy_data import DATASETS  # noqa: E402

from rf import euler_sample, rf_loss  # noqa: E402

DATASET = "eight_gaussians"


class VelocityNet(nn.Module):
    """MLP velocity field v(x, t) — the toy sibling of the image U-Net."""

    def __init__(self, hidden: int = 128, fourier: int = 16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(fourier) * 2.0)
        self.net = nn.Sequential(
            nn.Linear(2 + 2 * fourier, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        proj = t[:, None] * self.freqs[None, :] * 6.28318
        emb = torch.cat([proj.sin(), proj.cos()], dim=1)
        return self.net(torch.cat([x, emb], dim=1))


def train(steps: int = 4000, seed: int = 0, pairs=None) -> VelocityNet:
    """Train on random (x0, eps) pairing, or on fixed couples (re-flow)."""
    torch.manual_seed(seed)
    model = VelocityNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(1, steps + 1):
        if pairs is None:
            x0, eps = DATASETS[DATASET](512), None
        else:
            idx = torch.randint(0, pairs[0].size(0), (512,))
            x0, eps = pairs[0][idx], pairs[1][idx]
        loss = rf_loss(model, x0, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"step {step}/{steps} | loss {loss.item():.4f}", flush=True)
    return model


def straightness(traj) -> float:
    """1 = perfectly straight paths: chord length / arc length, averaged."""
    xs = torch.stack(traj)  # (steps+1, n, 2)
    chord = (xs[-1] - xs[0]).norm(dim=1)
    arc = (xs[1:] - xs[:-1]).norm(dim=2).sum(dim=0)
    return float((chord / arc).mean())


def plot_paths(traj, title, out_path, n_show=40):
    data = DATASETS[DATASET](800, generator=torch.Generator().manual_seed(0))
    fig, ax = ps.new_axes(6.4, 6.4)
    ax.scatter(data[:, 0], data[:, 1], s=5, color=ps.INK_MUTED, alpha=0.25,
               linewidths=0)
    xs = torch.stack(traj)
    for k in range(n_show):
        ax.plot(xs[:, k, 0], xs[:, k, 1], color=ps.SERIES[0], linewidth=1.2,
                alpha=0.7)
    lim = 5.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ps.finish(fig, ax, title, "", "", out_path)


def plot_fewstep(model, out_path):
    import matplotlib.pyplot as plt

    data = DATASETS[DATASET](1500, generator=torch.Generator().manual_seed(1))
    counts = (1, 2, 4, 16)
    fig, axes = plt.subplots(1, len(counts), figsize=(3.0 * len(counts), 3.2), dpi=100)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, n in zip(axes, counts):
        torch.manual_seed(7)
        x, _ = euler_sample(model, torch.randn(1500, 2), n)
        ax.set_facecolor(ps.SURFACE)
        ax.scatter(data[:, 0], data[:, 1], s=2, color=ps.INK_MUTED, alpha=0.2,
                   linewidths=0)
        ax.scatter(x[:, 0], x[:, 1], s=2, color=ps.SERIES[0], alpha=0.5, linewidths=0)
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-5.5, 5.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(ps.BASELINE)
        ax.set_title(f"{n} Euler step{'s' if n > 1 else ''}",
                     color=ps.INK_SECONDARY, fontsize=10)
    fig.suptitle("Rectified flow: few-step sampling", color=ps.INK, fontsize=12,
                 x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    model = train()
    (HERE / "checkpoints").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict()}, HERE / "checkpoints/rf_toy.pt")

    torch.manual_seed(3)
    # The RF prior at t = 1 is exactly N(0, I) — x_1 = eps by construction.
    x, traj = euler_sample(model, torch.randn(200, 2), 60,
                           return_trajectory=True)
    print(f"straightness (1 = straight lines): {straightness(traj):.3f}")
    plot_paths(traj, "Rectified-flow sampling paths (60 Euler steps)",
               out_dir / "rf_paths.png")
    plot_fewstep(model, out_dir / "few_step.png")
