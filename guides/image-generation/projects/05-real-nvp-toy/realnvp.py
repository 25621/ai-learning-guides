"""Real NVP on 2D toy data: a normalizing flow you can actually see.

A normalizing flow warps a simple base distribution (here a 2D standard Gaussian)
into the data distribution through a stack of *invertible* maps. Because every
map is invertible and its Jacobian determinant is cheap, the flow can report the
EXACT density of any point via change-of-variables:

    log p(x) = log N(z) + sum_layers log|det J|,   z = f(x)

Real NVP's trick for making that determinant cheap is the affine *coupling*
layer: split the coordinates in two, leave half untouched, and scale-and-shift
the other half using functions of the untouched half. The Jacobian is
triangular, so its log-det is just the sum of the scales — and inverting the
layer is trivial algebra.

Working in 2D lets us render the learned density as a heatmap and overlay real
samples, making "modeling a probability density" concrete.

    python realnvp.py      # ~1 min on CPU, no dataset download (toy data is synthetic)
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


# --------------------------------------------------------------------------- #
# Synthetic 2D datasets (no sklearn dependency).
# --------------------------------------------------------------------------- #
def two_moons(n, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, math.pi, n // 2)
    outer = np.stack([np.cos(t), np.sin(t)], 1)
    inner = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)
    x = np.concatenate([outer, inner], 0) + 0.06 * rng.standard_normal((n, 2))
    return normalize(x)


def swiss_roll(n, seed=0):
    rng = np.random.default_rng(seed)
    t = 1.5 * math.pi * (1 + 2 * rng.random(n))
    x = np.stack([t * np.cos(t), t * np.sin(t)], 1)
    x += 0.5 * rng.standard_normal((n, 2))
    return normalize(x)


def normalize(x):
    x = (x - x.mean(0)) / x.std(0)
    return torch.tensor(x, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Real NVP: affine coupling layers over a 2D vector.
# --------------------------------------------------------------------------- #
class Coupling(nn.Module):
    def __init__(self, mask, hidden=64):
        super().__init__()
        self.register_buffer("mask", mask)               # (2,) with a 1 and a 0
        self.scale = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2), nn.Tanh())             # bounded log-scale
        self.trans = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2))

    def forward(self, x):
        """x -> z, returns log|det J| (per sample)."""
        xb = x * self.mask
        s = self.scale(xb) * (1 - self.mask)
        t = self.trans(xb) * (1 - self.mask)
        z = xb + (1 - self.mask) * (x * torch.exp(s) + t)
        return z, s.sum(1)

    def inverse(self, z):
        zb = z * self.mask
        s = self.scale(zb) * (1 - self.mask)
        t = self.trans(zb) * (1 - self.mask)
        x = zb + (1 - self.mask) * ((z - t) * torch.exp(-s))
        return x


class RealNVP(nn.Module):
    def __init__(self, n_layers=6, hidden=64):
        super().__init__()
        masks = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        self.layers = nn.ModuleList(Coupling(masks[i % 2], hidden) for i in range(n_layers))

    def forward(self, x):
        logdet = torch.zeros(x.size(0))
        z = x
        for layer in self.layers:
            z, ld = layer(z)
            logdet = logdet + ld
        return z, logdet

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        z, logdet = self(x)
        base = -0.5 * (z ** 2).sum(1) - math.log(2 * math.pi)   # log N(z; 0, I) in 2D
        return base + logdet

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, 2)
        return self.inverse(z)


def train(model, data, steps=2500, lr=1e-3, bs=256, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(data), (bs,))
        loss = -model.log_prob(data[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  step {step}/{steps} | nll {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def density_grid(model, lim=2.8, res=200):
    xs = torch.linspace(-lim, lim, res)
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
    logp = model.log_prob(pts).reshape(res, res)
    return gx, gy, logp.exp()


def plot_panel(model, data, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gx, gy, dens = density_grid(model)
    samp = model.sample(1500).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    axes[0].scatter(data[:, 0], data[:, 1], s=4, color="#2a78d6", alpha=0.5)
    axes[0].set_title("training data", fontsize=11)
    axes[1].pcolormesh(gx, gy, dens, cmap="magma", shading="auto")
    axes[1].set_title("learned density  p(x)", fontsize=11)
    axes[2].scatter(samp[:, 0], samp[:, 1], s=4, color="#1baf7a", alpha=0.5)
    axes[2].set_title("samples  (Gaussian → inverse flow)", fontsize=11)
    for ax in axes:
        ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.suptitle(f"Real NVP on {name}", fontsize=13, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for name, gen in [("two moons", two_moons), ("swiss roll", swiss_roll)]:
        print(f"training on {name} ...")
        data = gen(3000)
        model = RealNVP()
        train(model, data, args.steps)

        # Invertibility check: x -> z -> x round-trip error.
        with torch.no_grad():
            z, _ = model(data[:512])
            recon = model.inverse(z)
            rt = (recon - data[:512]).abs().mean().item()
            nll = -model.log_prob(data).mean().item()
        print(f"  {name}: test nll {nll:.3f} | round-trip |x - f^-1(f(x))| = {rt:.2e}")
        results.append((name, nll, rt))

        slug = name.replace(" ", "_")
        plot_panel(model, data, name, OUT / f"{slug}.png")

    lines = ["dataset,nll,roundtrip_mae"]
    for name, nll, rt in results:
        lines.append(f"{name},{nll:.3f},{rt:.2e}")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
