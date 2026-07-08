"""Shared VQ tokenizer building blocks for the Phase-3 projects.

A small convolutional encoder/decoder for 32x32x3 CIFAR images that compresses to
an 8x8 grid of D-dimensional vectors, plus two vector quantizers:

  - `VectorQuantizer`       — the classic loss-based version (straight-through
    estimator + codebook & commitment losses), used by project 12;
  - `VectorQuantizerEMA`    — exponential-moving-average codebook updates plus
    dead-code re-initialization, used by project 13 to beat codebook collapse.

Projects 13/14/16/17 import from here via sys.path.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# Encoder / decoder (32x32x3  <->  8x8xD)
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(), nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(), nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, ch=64, D=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1), nn.ReLU(),     # 32 -> 16
            nn.Conv2d(ch, ch, 4, 2, 1),                   # 16 -> 8
            ResBlock(ch), ResBlock(ch),
            nn.Conv2d(ch, D, 1))                          # 8x8xD
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_ch=3, ch=64, D=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(D, ch, 1),
            ResBlock(ch), ResBlock(ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.ReLU(),   # 8 -> 16
            nn.ConvTranspose2d(ch, out_ch, 4, 2, 1), nn.Tanh())  # 16 -> 32, [-1,1]
    def forward(self, z):
        return self.net(z)


# --------------------------------------------------------------------------- #
# Vector quantizers
# --------------------------------------------------------------------------- #
def _quantize(z_e, codebook):
    """z_e: (B,D,H,W); codebook: (K,D). Returns (indices (B,H,W), flat (BHW,D))."""
    B, D, H, W = z_e.shape
    flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)
    d = (flat.pow(2).sum(1, keepdim=True)
         - 2 * flat @ codebook.t()
         + codebook.pow(2).sum(1))
    idx = d.argmin(1)
    return idx.view(B, H, W), flat


class VectorQuantizer(nn.Module):
    """Classic VQ-VAE quantizer: nearest-neighbour lookup with straight-through
    gradients, trained by codebook + commitment losses."""

    def __init__(self, K=256, D=32, commitment=0.25):
        super().__init__()
        self.codebook = nn.Embedding(K, D)
        self.codebook.weight.data.uniform_(-1.0 / K, 1.0 / K)
        self.K, self.beta = K, commitment

    def forward(self, z_e):
        idx, _ = _quantize(z_e, self.codebook.weight)
        z_q = self.codebook(idx).permute(0, 3, 1, 2)
        codebook_loss = (z_q - z_e.detach()).pow(2).mean()
        commit_loss = (z_q.detach() - z_e).pow(2).mean()
        loss = codebook_loss + self.beta * commit_loss
        z_q = z_e + (z_q - z_e).detach()                 # straight-through
        return z_q, loss, idx


class VectorQuantizerEMA(nn.Module):
    """EMA codebook updates + dead-code re-initialization — the standard cure for
    codebook collapse. Only the commitment loss trains the encoder; the codebook
    is updated by moving averages, not gradients."""

    def __init__(self, K=512, D=32, commitment=0.25, decay=0.99, dead_after=256,
                 reinit=True, eps=1e-5):
        super().__init__()
        self.K, self.D, self.beta, self.decay, self.eps = K, D, commitment, decay, eps
        self.reinit, self.dead_after = reinit, dead_after
        embed = torch.randn(K, D) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("stale", torch.zeros(K))     # steps since last use

    def forward(self, z_e):
        idx, flat = _quantize(z_e, self.embed)
        z_q = F.embedding(idx, self.embed).permute(0, 3, 1, 2)
        commit_loss = (z_q.detach() - z_e).pow(2).mean() * self.beta
        onehot = F.one_hot(idx.reshape(-1), self.K).type(flat.dtype)   # (N,K)

        if self.training:
            n = onehot.sum(0)                                          # per-code count
            self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
            embed_sum = onehot.t() @ flat                             # (K,D)
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            N = self.cluster_size.sum()
            cs = (self.cluster_size + self.eps) / (N + self.K * self.eps) * N
            self.embed.copy_(self.embed_avg / cs.unsqueeze(1))
            # dead-code re-init: codes unused for a while get overwritten with a
            # random current encoder output plus noise, so they land on real data.
            self.stale = torch.where(n > 0, torch.zeros_like(self.stale), self.stale + 1)
            if self.reinit:
                dead = self.stale > self.dead_after
                if dead.any():
                    pick = flat[torch.randint(0, flat.size(0), (int(dead.sum()),))]
                    self.embed[dead] = pick + 0.01 * torch.randn_like(pick)
                    self.embed_avg[dead] = self.embed[dead]
                    self.cluster_size[dead] = 1.0
                    self.stale[dead] = 0

        z_q = z_e + (z_q - z_e).detach()
        return z_q, commit_loss, idx


def perplexity(idx, K):
    """exp(entropy) of the code-usage distribution: ~1 when collapsed to one
    code, up to K when all codes are used equally."""
    counts = torch.bincount(idx.reshape(-1), minlength=K).float()
    p = counts / counts.sum()
    ent = -(p[p > 0] * p[p > 0].log()).sum()
    return float(ent.exp()), counts


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def cifar_tensor(data_dir, train=True, n=None):
    arr = np.load(Path(data_dir) / f"cifar10_{'train' if train else 'test'}.npz")
    x = torch.from_numpy(arr["images"]).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1  # [-1,1]
    return x if n is None else x[:n]


def cifar_loader(data_dir, bs=128, train=True):
    return DataLoader(TensorDataset(cifar_tensor(data_dir, train)), batch_size=bs,
                      shuffle=train, num_workers=0, drop_last=train)


def infinite(loader):
    while True:
        yield from loader
