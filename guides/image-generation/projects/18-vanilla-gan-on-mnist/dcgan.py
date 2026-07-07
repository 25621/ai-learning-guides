"""Shared DCGAN generator/discriminator for Phase 4.

One small conv architecture, reused with different losses and conditioning
across the phase: vanilla non-saturating GAN (18), WGAN-GP (19), a
class-conditional projection discriminator (20), and as the fast toy generator
for GAN inversion (22) and the FID head-to-head (23).

All images are 32x32 (MNIST is zero-padded from 28x28; CIFAR-10 is already
32x32), so one architecture serves every project in the phase.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    """z (and, if n_classes>0, a one-hot label) -> a 32x32 image in [-1, 1]."""

    def __init__(self, nz=100, ngf=64, nc=1, n_classes=0):
        super().__init__()
        self.nz = nz
        self.n_classes = n_classes
        in_dim = nz + n_classes
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, ngf * 4, 4, 1, 0, bias=False),  # 1 -> 4
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 4 -> 8
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 8 -> 16
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # 16 -> 32
            nn.Tanh(),
        )

    def forward(self, z, y=None):
        if self.n_classes > 0:
            y_onehot = F.one_hot(y, self.n_classes).float()
            z = torch.cat([z, y_onehot], dim=1)
        z = z.view(z.size(0), -1, 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    """32x32 image -> a raw real/fake score (no sigmoid; apply the loss outside).

    `norm='batch'` for the vanilla GAN, `norm='none'` for WGAN-GP (batchnorm
    would couple samples within a batch, which breaks the per-sample gradient
    penalty).
    """

    def __init__(self, nc=1, ndf=64, norm="batch"):
        super().__init__()
        Norm = nn.BatchNorm2d if norm == "batch" else nn.Identity
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),  # 32 -> 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), Norm(ndf * 2), nn.LeakyReLU(0.2, True),  # 16 -> 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), Norm(ndf * 4), nn.LeakyReLU(0.2, True),  # 8 -> 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),  # 4 -> 1
        )

    def forward(self, x):
        return self.net(x).view(-1)


class ProjectionDiscriminator(nn.Module):
    """Class-conditional critic: a class-agnostic score plus a projection term
    embed(y) . features(x) (Miyato & Koyama, 2018)."""

    def __init__(self, nc=3, ndf=64, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),  # 32 -> 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),  # 16 -> 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),  # 8 -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(ndf * 4, 1)
        self.embed = nn.Embedding(n_classes, ndf * 4)

    def forward(self, x, y):
        h = self.pool(self.features(x)).flatten(1)
        out = self.linear(h).squeeze(1)
        out = out + (self.embed(y) * h).sum(1)
        return out


def weights_init(m):
    """DCGAN paper init: N(0, 0.02) on conv/bn weights. Makes training reliable."""
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def mnist_loader(data_dir, batch_size, train=True):
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.Pad(2),  # 28x28 -> 32x32
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    ds = datasets.MNIST(data_dir, train=train, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                        num_workers=0, drop_last=True)


def infinite(loader):
    while True:
        yield from loader
