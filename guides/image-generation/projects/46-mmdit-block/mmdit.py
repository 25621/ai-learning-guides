"""An SD3-style MMDiT block, and a mini model that trains on MNIST.

The one architectural idea: TWO token streams — a context stream (text in
SD3; here, learned "caption" tokens derived from the class label) and an
image stream — that share a single JOINT attention. Compare the phase-7
U-Net, where image tokens attend to frozen text through a separate
cross-attention: in MMDiT both streams are first-class citizens. Each
stream keeps its own AdaLN modulation, its own qkv projections, and its own
MLP (text and image statistics differ too much to share weights); only the
attention itself is joint — queries, keys, and values of both streams are
concatenated, so text attends to image just as image attends to text.

Trained here with rectified flow (project 45's objective) — the same
pairing SD3 and Flux ship: MMDiT + flow matching.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "43-implement-dit-s-2"))
from unet import sinusoidal_embedding  # noqa: E402


class Stream(nn.Module):
    """Everything one modality owns inside an MMDiT block."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)


class MMDiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.img = Stream(dim)
        self.txt = Stream(dim)

    def _qkv(self, stream: Stream, x, mods):
        sa, ca, ga, sm, cm, gm = mods
        h = stream.norm1(x) * (1 + ca) + sa
        B, N, C = h.shape
        q, k, v = stream.qkv(h).reshape(B, N, 3, self.heads, C // self.heads) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        return q, k, v

    def forward(self, x_img, x_txt, c, return_attn: bool = False):
        mods_i = self.img.ada(c)[:, None, :].chunk(6, dim=-1)
        mods_t = self.txt.ada(c)[:, None, :].chunk(6, dim=-1)

        qi, ki, vi = self._qkv(self.img, x_img, mods_i)
        qt, kt, vt = self._qkv(self.txt, x_txt, mods_t)

        # JOINT attention: one softmax over the concatenated token axis.
        q = torch.cat([qi, qt], dim=2)
        k = torch.cat([ki, kt], dim=2)
        v = torch.cat([vi, vt], dim=2)
        attn_map = None
        if return_attn:
            attn = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
            attn = attn.softmax(dim=-1)
            out = attn @ v
            attn_map = attn
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(x_img.size(0), -1, qi.size(1) * qi.size(-1))
        n_img = x_img.size(1)
        h_img, h_txt = out[:, :n_img], out[:, n_img:]

        # Per-stream projection, gate, residual, MLP.
        x_img = x_img + mods_i[2] * self.img.proj(h_img)
        x_txt = x_txt + mods_t[2] * self.txt.proj(h_txt)
        h = self.img.norm2(x_img) * (1 + mods_i[4]) + mods_i[3]
        x_img = x_img + mods_i[5] * self.img.mlp(h)
        h = self.txt.norm2(x_txt) * (1 + mods_t[4]) + mods_t[3]
        x_txt = x_txt + mods_t[5] * self.txt.mlp(h)
        return (x_img, x_txt, attn_map) if return_attn else (x_img, x_txt)


class MiniMMDiT(nn.Module):
    """MMDiT sized for MNIST: the 'caption' is 2 learned tokens per class."""

    def __init__(self, image_size=28, patch=4, in_ch=1, dim=128, depth=5,
                 heads=4, num_classes=10, n_txt_tokens=2):
        super().__init__()
        self.patch, self.in_ch, self.dim = patch, in_ch, dim
        self.grid = image_size // patch
        self.n_txt = n_txt_tokens

        self.patch_embed = nn.Conv2d(in_ch, dim, patch, stride=patch)
        self.pos_emb = nn.Parameter(torch.randn(1, self.grid**2, dim) * 0.02)
        self.txt_emb = nn.Embedding(num_classes, n_txt_tokens * dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                      nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(MMDiTBlock(dim, heads) for _ in range(depth))

        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.final_proj = nn.Linear(dim, patch * patch * in_ch)
        for m in (self.final_ada[-1], self.final_proj):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t, y, return_attn: bool = False):
        B = x.size(0)
        img = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_emb
        txt = self.txt_emb(y).reshape(B, self.n_txt, self.dim)
        c = self.time_mlp(sinusoidal_embedding(1000.0 * t, self.dim))

        attn_last = None
        for i, block in enumerate(self.blocks):
            if return_attn and i == len(self.blocks) - 1:
                img, txt, attn_last = block(img, txt, c, return_attn=True)
            else:
                img, txt = block(img, txt, c)

        img = self.final_norm(img)
        shift, scale = self.final_ada(c)[:, None, :].chunk(2, dim=-1)
        img = img * (1 + scale) + shift
        out = self.final_proj(img)
        out = out.reshape(B, self.grid, self.grid, self.patch, self.patch, self.in_ch)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(
            B, self.in_ch, self.grid * self.patch, self.grid * self.patch)
        return (out, attn_last) if return_attn else out


if __name__ == "__main__":
    model = MiniMMDiT()
    x = torch.randn(2, 1, 28, 28)
    t = torch.rand(2)
    y = torch.tensor([3, 7])
    print("params:", sum(p.numel() for p in model.parameters()))
    print("output:", model(x, t, y).shape)
