"""Shared autoencoder / VAE building blocks for the Phase-2 projects.

A small convolutional encoder/decoder pair for 28x28 single-channel images, plus
an `AE` (deterministic) and a `VAE` (variational) that reuse them. Projects 07
(vanilla VAE), 08 (beta-VAE) and 11 (latent traversal) import from here via
sys.path so every Phase-2 model shares one architecture.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Encoder(nn.Module):
    """28x28 -> (zdim * out_mult). out_mult=2 packs (mu, logvar) for a VAE."""

    def __init__(self, in_ch=1, base=32, zdim=16, out_mult=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1), nn.ReLU(),       # 28 -> 14
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.ReLU(),    # 14 -> 7
            nn.Flatten(),
        )
        self.fc = nn.Linear(base * 2 * 7 * 7, zdim * out_mult)

    def forward(self, x):
        return self.fc(self.body(x))


class Decoder(nn.Module):
    """zdim -> 28x28 in [0,1] (Bernoulli decoder, sigmoid output)."""

    def __init__(self, out_ch=1, base=32, zdim=16):
        super().__init__()
        self.fc = nn.Linear(zdim, base * 2 * 7 * 7)
        self.base = base
        self.body = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.ReLU(),   # 7 -> 14
            nn.ConvTranspose2d(base, out_ch, 4, 2, 1), nn.Sigmoid(),  # 14 -> 28
        )

    def forward(self, z):
        h = self.fc(z).view(-1, self.base * 2, 7, 7)
        return self.body(h)


class AE(nn.Module):
    def __init__(self, zdim=32, base=32):
        super().__init__()
        self.enc = Encoder(zdim=zdim, base=base, out_mult=1)
        self.dec = Decoder(zdim=zdim, base=base)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z

    def loss(self, x):
        xhat, _ = self(x)
        return F.binary_cross_entropy(xhat, x, reduction="none").sum((1, 2, 3)).mean()


class VAE(nn.Module):
    def __init__(self, zdim=16, base=32):
        super().__init__()
        self.zdim = zdim
        self.enc = Encoder(zdim=zdim, base=base, out_mult=2)
        self.dec = Decoder(zdim=zdim, base=base)

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)          # gradients flow through mu, std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

    def loss(self, x, beta=1.0):
        """Returns (total, recon_bits_per_dim-free recon, kl) all per-image."""
        xhat, mu, logvar = self(x)
        recon = F.binary_cross_entropy(xhat, x, reduction="none").sum((1, 2, 3)).mean()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        return recon + beta * kl, recon, kl

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.zdim)
        return self.dec(z)


def mnist_loader(data_dir, bs=128, train=True):
    tf = transforms.Compose([transforms.ToTensor()])       # [0,1]
    ds = datasets.MNIST(str(data_dir), train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=2, drop_last=train)


def infinite(loader):
    while True:
        yield from loader
