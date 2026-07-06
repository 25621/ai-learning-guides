"""Measure the quality-vs-steps trade-off of Euler and Heun on one DDPM.

Everything starts from the SAME noise; the reference solution is a very fine
Heun run. The error of each (solver, step-count) pair is the RMSE between its
endpoint and the reference endpoint — a direct measurement of solver
truncation error, plotted against NFE (number of model evaluations).

Run (after training a DDPM into checkpoints/mnist_ddpm.pt):
    python compare_solvers.py
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
import plot_style as ps  # noqa: E402
from diffusion import GaussianDiffusion  # noqa: E402
from unet import UNet  # noqa: E402

from solvers import DDPMDenoiser, ode_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints/mnist_ddpm.pt"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = UNet().to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule=ckpt["schedule"], device=device)
    den = DDPMDenoiser(model, diffusion)

    torch.manual_seed(args.seed)
    x_init = den.sigma_max * torch.randn(args.n, 1, 28, 28, device=device)

    reference, _ = ode_sample(den, x_init, n_steps=200, method="heun")

    step_counts = (5, 10, 20, 50)
    rows, grid_rows = [("method", "steps", "nfe", "rmse_vs_reference")], [reference]
    curves = {"euler": ([], []), "heun": ([], [])}
    for method in ("euler", "heun"):
        for n_steps in step_counts:
            x, nfe = ode_sample(den, x_init, n_steps=n_steps, method=method)
            rmse = (x - reference).pow(2).mean().sqrt().item()
            rows.append((method, n_steps, nfe, f"{rmse:.4f}"))
            curves[method][0].append(nfe)
            curves[method][1].append(rmse)
            grid_rows.append(x)
            print(f"{method:>5} {n_steps:3d} steps ({nfe:3d} NFE) | RMSE {rmse:.4f}",
                  flush=True)

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "solver_errors.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # Grid: reference on top, then euler 5/10/20/50, then heun 5/10/20/50.
    save_image(
        F.interpolate(torch.cat(grid_rows).clamp(-1, 1), scale_factor=2, mode="nearest"),
        out_dir / "solver_grids.png", nrow=args.n, normalize=True, value_range=(-1, 1),
    )

    # Convergence plot: error vs NFE, log-log. Slope ~ -1 for Euler, ~ -2 for Heun.
    fig, ax = ps.new_axes(6.4, 4.2)
    for method, color in (("euler", ps.SERIES[0]), ("heun", ps.SERIES[1])):
        nfes, errs = curves[method]
        ax.plot(nfes, errs, color=color, linewidth=2, marker="o", markersize=5)
        ax.annotate(method, (nfes[-1], errs[-1]), color=ps.INK_SECONDARY,
                    fontsize=10, fontweight="bold", xytext=(8, 0),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.5)
    ps.finish(fig, ax, "Solver error vs model evaluations (log-log)",
              "NFE (model evaluations)", "RMSE vs fine reference",
              out_dir / "error_vs_nfe.png")


if __name__ == "__main__":
    main()
