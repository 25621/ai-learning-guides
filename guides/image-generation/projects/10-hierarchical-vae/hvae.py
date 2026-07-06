"""Hierarchical VAE on CIFAR-10: does splitting the latent across scales help?

A flat VAE compresses an image into one latent grid. A hierarchical VAE uses two
latent grids at different resolutions — a coarse 4x4 that carries global shape and
color, and a finer 8x8 that fills in local detail — and the decoder combines
both. Natural images have structure at many scales at once, so dividing the
labor should help.

We test that head-to-head: a two-level VAE vs. a flat VAE with the *same* total
latent budget (512 numbers each), trained identically, compared on
reconstruction error and samples.

    python hvae.py --data-dir data      # ~9 min on CPU (two CIFAR VAEs)

Loads CIFAR-10 from cifar10_train.npz / cifar10_test.npz (NHWC uint8) in data-dir.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
OUT = HERE / "outputs"

TC, BC, FC = 16, 4, 8   # top 4x4x16=256, bottom 8x8x4=256 (=512); flat 8x8x8=512


def cifar_loader(data_dir, bs=128, train=True):
    arr = np.load(Path(data_dir) / f"cifar10_{'train' if train else 'test'}.npz")
    x = torch.from_numpy(arr["images"]).float().permute(0, 3, 1, 2) / 255.0   # (N,3,32,32) [0,1]
    ds = TensorDataset(x)
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=2, drop_last=train)


def kl_std_normal(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).flatten(1).sum(1).mean()


def reparam(mu, logvar):
    return mu + (0.5 * logvar).exp() * torch.randn_like(mu)


class FlatVAE(nn.Module):
    """Single 8x8 latent grid."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),        # 16
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),       # 8
            nn.Conv2d(64, FC * 2, 3, 1, 1))              # 8x8, (mu,logvar)
        self.dec = nn.Sequential(
            nn.Conv2d(FC, 64, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),   # 16
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()) # 32

    def forward(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=1)
        z = reparam(mu, logvar)
        return self.dec(z), kl_std_normal(mu, logvar)

    @torch.no_grad()
    def sample(self, n):
        return self.dec(torch.randn(n, FC, 8, 8))


class HierVAE(nn.Module):
    """Two latents: top 4x4 (coarse) + bottom 8x8 (fine); decoder fuses both."""
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU())      # 16
        self.e2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU())     # 8
        self.e3 = nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU())     # 4
        self.top = nn.Conv2d(64, TC * 2, 3, 1, 1)                          # 4x4
        self.top_up = nn.Sequential(
            nn.ConvTranspose2d(TC, 64, 4, 2, 1), nn.ReLU())                # 4 -> 8
        self.bottom = nn.Conv2d(64 + 64, BC * 2, 3, 1, 1)                  # 8x8 from (top_feat, h2)
        self.dec = nn.Sequential(
            nn.Conv2d(BC + 64, 64, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),   # 16
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()) # 32

    def forward(self, x):
        h2 = self.e2(self.e1(x))                              # 8x8x64
        h3 = self.e3(h2)                                      # 4x4x64
        mu_t, lv_t = self.top(h3).chunk(2, dim=1)
        z_top = reparam(mu_t, lv_t)                           # 4x4xTC
        top_feat = self.top_up(z_top)                         # 8x8x64
        mu_b, lv_b = self.bottom(torch.cat([top_feat, h2], 1)).chunk(2, dim=1)
        z_bot = reparam(mu_b, lv_b)                           # 8x8xBC
        xhat = self.dec(torch.cat([z_bot, top_feat], 1))
        kl = kl_std_normal(mu_t, lv_t) + kl_std_normal(mu_b, lv_b)
        return xhat, kl

    @torch.no_grad()
    def sample(self, n):
        z_top = torch.randn(n, TC, 4, 4)
        top_feat = self.top_up(z_top)
        z_bot = torch.randn(n, BC, 8, 8)
        return self.dec(torch.cat([z_bot, top_feat], 1))


def train(model, data_dir, steps, beta=1.0, lr=1e-3):
    loader = cifar_loader(data_dir, 128, True)
    it = iter(loader); opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            (x,) = next(it)
        except StopIteration:
            it = iter(loader); (x,) = next(it)
        xhat, kl = model(x)
        recon = F.binary_cross_entropy(xhat, x, reduction="none").sum((1, 2, 3)).mean()
        loss = recon + beta * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 400 == 0:
            print(f"    step {step}/{steps} | recon {recon.item():.1f} | kl {kl.item():.1f} | "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)


@torch.no_grad()
def test_mse(model, loader):
    tot, n = 0.0, 0
    for (x,) in loader:
        xhat, _ = model(x)
        tot += F.mse_loss(xhat, x, reduction="sum").item(); n += x.numel()
    return tot / n


def show(imgs):
    return imgs.clamp(0, 1).permute(0, 2, 3, 1).numpy()


def plot_recon(models, loader, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (x,) = next(iter(loader)); x = x[:10]
    rows = [("input", x)]
    with torch.no_grad():
        for name, m in models.items():
            rows.append((name, m(x)[0]))
    R = len(rows)
    fig, axes = plt.subplots(R, 10, figsize=(10, R * 1.05), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for r, (name, imgs) in enumerate(rows):
        vis = show(imgs)
        for c in range(10):
            axes[r][c].imshow(vis[c]); axes[r][c].axis("off")
        fig.text(0.09, 1 - (r + 0.5) / R * 0.9 - 0.02, name, ha="right",
                 fontsize=9, color="#0b0b0b")
    fig.suptitle("CIFAR reconstructions: flat vs hierarchical (same latent budget)",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0.1, 0, 1, 0.94])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_samples(models, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.manual_seed(1)
    R = len(models)
    fig, axes = plt.subplots(R, 10, figsize=(10, R * 1.05), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    with torch.no_grad():
        for r, (name, m) in enumerate(models.items()):
            vis = show(m.sample(10))
            for c in range(10):
                axes[r][c].imshow(vis[c]); axes[r][c].axis("off")
            fig.text(0.09, 1 - (r + 0.5) / R * 0.9 - 0.02, name, ha="right",
                     fontsize=9, color="#0b0b0b")
    fig.suptitle("Samples z ~ N(0,I) (CIFAR VAEs are blurry — compare structure, not sharpness)",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=[0.1, 0, 1, 0.93])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--steps", type=int, default=2500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    models = {}
    for name, cls in [("flat", FlatVAE), ("hierarchical", HierVAE)]:
        torch.manual_seed(0)
        m = cls()
        print(f"training {name} VAE ({sum(p.numel() for p in m.parameters()):,} params) ...")
        train(m, args.data_dir, args.steps)
        m.eval(); models[name] = m

    test = cifar_loader(args.data_dir, 200, train=False)
    mses = {name: test_mse(m, test) for name, m in models.items()}
    for name, v in mses.items():
        print(f"  {name:12s} test reconstruction MSE: {v:.5f}")
    gain = (mses["flat"] - mses["hierarchical"]) / mses["flat"] * 100
    print(f"  hierarchical improves reconstruction MSE by {gain:.1f}%")

    plot_recon(models, test, OUT / "reconstructions.png")
    plot_samples(models, OUT / "samples.png")
    (OUT / "results.csv").write_text(
        "model,latent_budget,test_mse\n"
        f"flat,512,{mses['flat']:.5f}\n"
        f"hierarchical,512,{mses['hierarchical']:.5f}\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
