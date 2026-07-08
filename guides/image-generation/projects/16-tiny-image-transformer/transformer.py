"""A minimal pre-norm transformer block, shared by the autoregressive GPT
(project 16) and the bidirectional masked model (project 17)."""

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_mult=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_mult), nn.GELU(), nn.Linear(dim * mlp_mult, dim))

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


def causal_mask(n, device="cpu"):
    """(n, n) float mask: 0 on/below diagonal, -inf above (block the future)."""
    return torch.triu(torch.full((n, n), float("-inf"), device=device), diagonal=1)
