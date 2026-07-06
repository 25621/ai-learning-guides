"""All README figures for the probability flow ODE project.

Run (after project 30's train_score.py):
    python make_figures.py
"""

import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "30-score-matching-from-scratch"))
import plot_style as ps  # noqa: E402
from toy_data import DATASETS  # noqa: E402

from pf_ode import load_score_model, log_likelihood, ode_sample, sde_sample  # noqa: E402

DATASET = "eight_gaussians"


def plot_trajectories(model, out_path):
    """The signature figure: smooth ODE paths vs jagged SDE paths, same model."""
    n = 6
    _, ode_traj = ode_sample(model, n=n, n_steps=80, seed=3, return_trajectory=True)
    _, sde_traj = sde_sample(model, n=n, n_steps=300, seed=3, return_trajectory=True)
    data = DATASETS[DATASET](800, generator=torch.Generator().manual_seed(0))

    fig, ax = ps.new_axes(7.4, 7.4)
    ax.scatter(data[:, 0], data[:, 1], s=5, color=ps.INK_MUTED, alpha=0.25, linewidths=0)
    sde_xy = torch.stack(sde_traj)  # (steps, n, 2)
    ode_xy = torch.stack(ode_traj)
    for k in range(n):
        (sde_line,) = ax.plot(sde_xy[:, k, 0], sde_xy[:, k, 1], color=ps.SERIES[1],
                              linewidth=0.6, alpha=0.45)
    for k in range(n):
        (ode_line,) = ax.plot(ode_xy[:, k, 0], ode_xy[:, k, 1], color=ps.SERIES[0],
                              linewidth=1.8, alpha=0.9)
    ax.legend([ode_line, sde_line], ["probability flow ODE", "reverse SDE"],
              loc="upper left", framealpha=0.92, fontsize=9)
    lim = 9
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ps.finish(fig, ax, "Two samplers, one model: deterministic vs stochastic paths",
              "", "", out_path)


def plot_samples(model, out_path):
    samples, _ = ode_sample(model, n=2000, n_steps=80, seed=7)
    data = DATASETS[DATASET](2000, generator=torch.Generator().manual_seed(1))
    fig, ax = ps.new_axes(6.0, 6.0)
    ax.scatter(data[:, 0], data[:, 1], s=5, color=ps.INK_MUTED, alpha=0.3,
               linewidths=0, label="true data")
    ax.scatter(samples[:, 0], samples[:, 1], s=5, color=ps.SERIES[0], alpha=0.55,
               linewidths=0, label="ODE samples (80 Heun steps)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9, markerscale=3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ps.finish(fig, ax, "Probability-flow ODE samples vs true data", "", "", out_path)


def plot_density(model, out_path):
    """Exact log p(x) on a grid — a picture of what the model believes."""
    import matplotlib.pyplot as plt

    lim, res = 5.5, 64
    grid = torch.linspace(-lim, lim, res)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    pts = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    logp = log_likelihood(model, pts, n_steps=60).view(res, res)
    data = DATASETS[DATASET](600, generator=torch.Generator().manual_seed(2))

    fig, ax = ps.new_axes(6.6, 6.0)
    im = ax.imshow(logp.clamp(min=-9), extent=(-lim, lim, -lim, lim), origin="lower",
                   cmap="Blues", aspect="equal")
    ax.scatter(data[:, 0], data[:, 1], s=3, color=ps.INK, alpha=0.4, linewidths=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("log p(x)  (nats)", color=ps.INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=ps.INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    ax.grid(False)
    ps.finish(fig, ax, "Exact model log-density (via the PF-ODE), data overlaid",
              "", "", out_path)


def nll_table(model, out_path):
    """Held-out data vs the same points rotated 22.5 degrees — which lands
    them exactly BETWEEN the eight modes. Same radius, same scale, wrong
    place: the likelihood should notice."""
    import math

    held_out = DATASETS[DATASET](1000, generator=torch.Generator().manual_seed(3))
    a = math.pi / 8
    rot = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    rotated = held_out @ rot.T
    rows = [("points", "mean NLL (nats)")]
    for name, pts in (("held-out data", held_out),
                      ("same points rotated 22.5 deg (between modes)", rotated)):
        nll = -log_likelihood(model, pts, n_steps=80).mean().item()
        rows.append((name, f"{nll:.2f}"))
        print(f"{name}: mean NLL = {nll:.2f} nats", flush=True)
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    model = load_score_model(DATASET)
    plot_trajectories(model, out_dir / "ode_vs_sde_paths.png")
    plot_samples(model, out_dir / "ode_samples.png")
    plot_density(model, out_dir / "log_density.png")
    nll_table(model, out_dir / "nll.csv")
