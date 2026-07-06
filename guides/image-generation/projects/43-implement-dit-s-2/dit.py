"""A Diffusion Transformer (Peebles & Xie 2022) at teaching scale.

The full recipe, faithfully: patchify the image into tokens, run transformer
blocks whose LayerNorms are modulated by the conditioning vector (AdaLN-Zero),
and unpatchify a per-token prediction back into an image. The reference
DiT-S/2 is 12 blocks of dim 384 on 32x32x4 latents; DIT_MINI below is the
same machine sized for a CPU (6 blocks, dim 128, patch 4 on MNIST pixels).

Attention is written out by hand (qkv + scaled_dot_product_attention) rather
than nn.MultiheadAttention so that project 44 can inject rotary position
embeddings into q and k, and project 46 can reuse the pieces for joint
text-image attention.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "24-ddpm-on-mnist"))
from unet import sinusoidal_embedding  # noqa: E402

DIT_MINI = dict(image_size=28, patch=4, in_ch=1, dim=128, depth=6, heads=4)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, rope=None):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rope = rope  # optional callable applied to q and k (project 44)

    def forward(self, x, return_attn: bool = False):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads) \
            .permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, N, head_dim)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        if return_attn:
            attn = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
            attn = attn.softmax(dim=-1)
            out = attn @ v
        else:
            attn = None
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return (self.proj(out), attn) if return_attn else self.proj(out)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, rope=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads, rope=rope)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # (shift, scale, gate) x {attn, mlp}, zero-init: block starts as identity.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        sa, ca, ga, sm, cm, gm = self.ada(c)[:, None, :].chunk(6, dim=-1)
        x = x + ga * self.attn(self.norm1(x) * (1 + ca) + sa)
        x = x + gm * self.mlp(self.norm2(x) * (1 + cm) + sm)
        return x


class DiT(nn.Module):
    def __init__(self, image_size=28, patch=4, in_ch=1, dim=128, depth=6,
                 heads=4, num_classes=10, rope=None, learned_pos=True):
        super().__init__()
        self.patch, self.in_ch, self.dim = patch, in_ch, dim
        self.grid = image_size // patch
        n_tokens = self.grid * self.grid

        self.patch_embed = nn.Conv2d(in_ch, dim, patch, stride=patch)
        self.pos_emb = (
            nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02) if learned_pos else None
        )
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.label_emb = nn.Embedding(num_classes, dim) if num_classes else None

        self.blocks = nn.ModuleList(
            DiTBlock(dim, heads, rope=rope) for _ in range(depth)
        )
        # Final layer: one more AdaLN modulation, then project to patch pixels.
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.final_proj = nn.Linear(dim, patch * patch * in_ch)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def cond(self, t, y=None):
        c = self.time_mlp(sinusoidal_embedding(t, self.dim))
        if self.label_emb is not None and y is not None:
            c = c + self.label_emb(y)
        return c

    def unpatchify(self, x, grid_h: int, grid_w: int):
        B = x.size(0)
        p, ch = self.patch, self.in_ch
        x = x.reshape(B, grid_h, grid_w, p, p, ch)
        x = x.permute(0, 5, 1, 3, 2, 4)  # B, C, gh, p, gw, p
        return x.reshape(B, ch, grid_h * p, grid_w * p)

    def forward(self, x, t, y=None):
        B, _, H, W = x.shape
        gh, gw = H // self.patch, W // self.patch
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, dim)
        if self.pos_emb is not None:
            pos = self.pos_emb
            if pos.size(1) != tokens.size(1):
                # Sampling at a resolution the learned table wasn't built for:
                # bilinear interpolation, the standard ViT patch. Project 44
                # shows what this costs versus RoPE.
                g = int(pos.size(1) ** 0.5)
                pos = pos.transpose(1, 2).reshape(1, self.dim, g, g)
                pos = F.interpolate(pos, size=(gh, gw), mode="bilinear",
                                    align_corners=False)
                pos = pos.flatten(2).transpose(1, 2)
            tokens = tokens + pos
        c = self.cond(t, y)
        for block in self.blocks:
            tokens = block(tokens, c)
        tokens = self.final_norm(tokens)
        shift, scale = self.final_ada(c)[:, None, :].chunk(2, dim=-1)
        tokens = tokens * (1 + scale) + shift
        return self.unpatchify(self.final_proj(tokens), gh, gw)


if __name__ == "__main__":
    model = DiT(**DIT_MINI)
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 300, (2,))
    y = torch.tensor([3, 7])
    print("params:", sum(p.numel() for p in model.parameters()))
    print("output:", model(x, t, y).shape)
