"""GAN inversion, the controlled comparison: optimization vs a learned encoder,
on the same small MNIST DCGAN (so both methods pay the exact same per-step
compute cost and the only variable is the strategy).

StyleGAN2 inversion (`stylegan_invert.py`) is too expensive on CPU to also
train an encoder for — generating enough (image, latent) training pairs at
1024x1024 would take longer than this whole guide's time budget. That's
exactly the point this script makes concrete: optimization pays its cost
*every time* (a fresh 100+ step search per image), while a learned encoder
pays a large one-time training cost and then inverts any new image in a
single forward pass. At toy scale we can afford to train the encoder and
measure both strategies head to head.

    python toy_invert.py --data-dir data   # ~1.5 min on CPU
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

sys.path.insert(0, str(HERE.parent / "18-vanilla-gan-on-mnist"))
from dcgan import Generator, Discriminator, infinite, mnist_loader, weights_init  # noqa: E402

NZ = 100
N_TARGETS = 5
OPT_STEPS = 150
ENCODER_STEPS = 600


class Encoder(nn.Module):
    """Mirror of the Discriminator, but regresses to a z vector instead of a score."""

    def __init__(self, nz=NZ, nc=1, nef=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(nef * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(nef * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nef * 4, nz, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)


def train_generator(data_dir, seed=0, steps=400):
    torch.manual_seed(seed)
    loader = infinite(mnist_loader(data_dir, batch_size=128))
    G = Generator(nz=NZ, nc=1); G.apply(weights_init)
    D = Discriminator(nc=1); D.apply(weights_init)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    for step in range(steps):
        x_real, _ = next(loader)
        z = torch.randn(x_real.size(0), NZ)
        with torch.no_grad():
            x_fake = G(z)
        d_real, d_fake = D(x_real), D(x_fake)
        loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                  + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        z = torch.randn(128, NZ)
        d_fake = D(G(z))
        loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G


def train_encoder(G, steps=ENCODER_STEPS, seed=1):
    """Synthetic-pair training: sample z, generate G(z), regress the encoder
    back to the z that made it. Never needs to backprop through G."""
    torch.manual_seed(seed)
    E = Encoder()
    opt = torch.optim.Adam(E.parameters(), lr=1e-3)
    t0 = time.time()
    for step in range(steps):
        z = torch.randn(128, NZ)
        with torch.no_grad():
            img = G(z)
        z_pred = E(img)
        loss = F.mse_loss(z_pred, z)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 150 == 0:
            print(f"  encoder step {step} z-mse={loss.item():.3f}")
    print(f"  encoder trained in {time.time() - t0:.1f}s")
    E.eval()
    return E


@torch.no_grad()
def get_targets(data_dir, n, seed=2):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    imgs = torch.stack([ds[i][0] for i in idx])
    return imgs


def invert_by_optimization(G, target, steps=OPT_STEPS):
    z = torch.zeros(1, NZ, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.05)
    t0 = time.time()
    history = []
    for step in range(steps):
        opt.zero_grad()
        img = G(z)
        loss = F.mse_loss(img, target)
        loss.backward()
        opt.step()
        history.append(loss.item())
    elapsed = time.time() - t0
    with torch.no_grad():
        recon = G(z)
    return recon.detach(), elapsed, history


@torch.no_grad()
def invert_by_encoder(E, G, target):
    t0 = time.time()
    z_pred = E(target)
    recon = G(z_pred)
    elapsed = time.time() - t0
    return recon, elapsed


def run(data_dir):
    t0 = time.time()
    print("training the toy generator...")
    G = train_generator(data_dir)
    print(f"generator ready ({time.time() - t0:.0f}s)")

    print("training the encoder (synthetic pairs only, never touches a real image)...")
    E = train_encoder(G)
    print(f"encoder ready ({time.time() - t0:.0f}s)")

    targets = get_targets(data_dir, N_TARGETS)

    rows = []
    opt_recons, enc_recons, opt_curves = [], [], []
    for i in range(N_TARGETS):
        target = targets[i:i + 1]
        opt_recon, opt_time, history = invert_by_optimization(G, target)
        enc_recon, enc_time = invert_by_encoder(E, G, target)
        opt_mse = F.mse_loss(opt_recon, target).item()
        enc_mse = F.mse_loss(enc_recon, target).item()
        opt_recons.append(opt_recon[0, 0].numpy())
        enc_recons.append(enc_recon[0, 0].numpy())
        opt_curves.append(history)
        rows.append((i, opt_mse, opt_time, enc_mse, enc_time))
        print(f"target {i}: optimization mse={opt_mse:.4f} ({opt_time*1000:.0f}ms, {OPT_STEPS} steps) | "
              f"encoder mse={enc_mse:.4f} ({enc_time*1000:.1f}ms, 1 forward)")

    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "toy_invert.npz",
              targets=targets[:, 0].numpy(), opt_recons=np.stack(opt_recons),
              enc_recons=np.stack(enc_recons), opt_curves=np.array(opt_curves),
              rows=np.array(rows))

    with open(OUT / "toy_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "optimization_mse", "optimization_time_s", "encoder_mse", "encoder_time_s"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.3f}", f"{r[3]:.4f}", f"{r[4]:.4f}"])
    print(f"wrote {OUT / 'toy_results.csv'}")
    print(f"total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()
    run(args.data_dir)
