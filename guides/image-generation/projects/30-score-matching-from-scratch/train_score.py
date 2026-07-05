"""Denoising score matching on 2D toy data.

The whole objective in three lines: perturb a clean point with noise of a
random scale sigma, and train the network so that sigma * s(x + sigma*eps,
sigma) points back along -eps. Vincent (2011) proved this regression target
equals the true score of the noised distribution — no density, no partition
function, ever.

Run:
    python train_score.py                 # trains both toy datasets, ~1 min CPU
"""

import argparse
from pathlib import Path

import torch

from score_net import ScoreNet
from toy_data import DATASETS

HERE = Path(__file__).resolve().parent

SIGMA_MIN, SIGMA_MAX = 0.05, 8.0  # covers "barely blurred" to "pure cloud"


def dsm_loss(model: ScoreNet, x0: torch.Tensor) -> torch.Tensor:
    B = x0.size(0)
    # Log-uniform sigmas: every noise scale gets equal training attention.
    u = torch.rand(B)
    sigma = SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** u
    eps = torch.randn_like(x0)
    x_noisy = x0 + sigma[:, None] * eps
    # sigma * score should reproduce -eps  (the sigma factor balances scales)
    return ((sigma[:, None] * model(x_noisy, sigma) + eps) ** 2).mean()


def train(dataset: str, steps: int, out_path: Path, seed: int = 0):
    torch.manual_seed(seed)
    model = ScoreNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(1, steps + 1):
        x0 = DATASETS[dataset](512)
        loss = dsm_loss(model, x0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"{dataset} step {step}/{steps} | loss {loss.item():.4f}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "dataset": dataset,
                "sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX}, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    for name in DATASETS:
        train(name, args.steps, HERE / f"checkpoints/{name}.pt")
