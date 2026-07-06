"""A small U-Net for DDPM noise prediction.

The network takes a noised image x_t and a timestep t and predicts the noise
that was mixed into it. Architecture follows Ho et al. 2020 at toy scale:

- sinusoidal timestep embedding -> 2-layer MLP
- down/up blocks are residual blocks with GroupNorm + SiLU
- the time embedding enters every residual block as a FiLM (scale, shift)
  modulation of the second GroupNorm
- self-attention at the lowest resolution, where it is cheap
- skip connections carry fine detail from the encoder to the decoder
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer timesteps (B,) to smooth (B, dim) vectors, transformer-style."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> conv, twice; time embedding applied as FiLM."""

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # One linear layer emits both the scale and the shift of the FiLM.
        self.temb_proj = nn.Linear(temb_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.temb_proj(F.silu(temb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention over spatial positions (used at low res only)."""

    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(dim=1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)  # (B, HW, HW)
        h = (v @ attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """Noise-prediction U-Net: eps_theta(x_t, t).

    `ch_mults` defines the channel width at each resolution level; the image is
    downsampled between levels. `attn_resolutions` lists the spatial sizes at
    which self-attention runs (keep it to the smallest sizes).
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        ch_mults: tuple = (1, 2, 2),
        num_res_blocks: int = 1,
        attn_resolutions: tuple = (7,),
        image_size: int = 28,
        temb_dim: int = 128,
    ):
        super().__init__()
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        chs = [base_ch * m for m in ch_mults]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        # --- encoder ---
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        res = image_size
        ch = chs[0]
        self.skip_chs = [ch]  # in_conv output is the first skip
        for level, out_ch in enumerate(chs):
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, temb_dim))
                attns.append(SelfAttention(out_ch) if res in attn_resolutions else nn.Identity())
                ch = out_ch
                self.skip_chs.append(ch)
            self.down_blocks.append(blocks)
            self.down_attns.append(attns)
            if level < len(chs) - 1:
                self.downsamples.append(Downsample(ch))
                self.skip_chs.append(ch)
                res //= 2
            else:
                self.downsamples.append(nn.Identity())

        # --- middle ---
        self.mid_block1 = ResBlock(ch, ch, temb_dim)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, temb_dim)

        # --- decoder (mirror of the encoder, consuming skips) ---
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        skip_chs = list(self.skip_chs)
        for level, out_ch in reversed(list(enumerate(chs))):
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResBlock(ch + skip_chs.pop(), out_ch, temb_dim))
                attns.append(SelfAttention(out_ch) if res in attn_resolutions else nn.Identity())
                ch = out_ch
            self.up_blocks.append(blocks)
            self.up_attns.append(attns)
            if level > 0:
                self.upsamples.append(Upsample(ch))
                res *= 2
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)  # start by predicting zero noise
        nn.init.zeros_(self.out_conv.bias)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(sinusoidal_embedding(t, self.temb_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward_with_temb(x, self.time_embedding(t))

    def forward_with_temb(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Run the U-Net given an already-built conditioning vector. Subclasses
        (e.g. a class-conditional model) add their extra signals to `temb`
        before calling this — every ResBlock then sees them through its FiLM
        modulation."""
        h = self.in_conv(x)
        skips = [h]
        for level, (blocks, attns) in enumerate(zip(self.down_blocks, self.down_attns)):
            for block, attn in zip(blocks, attns):
                h = attn(block(h, temb))
                skips.append(h)
            if not isinstance(self.downsamples[level], nn.Identity):
                h = self.downsamples[level](h)
                skips.append(h)

        h = self.mid_block2(self.mid_attn(self.mid_block1(h, temb)), temb)

        for blocks, attns, up in zip(self.up_blocks, self.up_attns, self.upsamples):
            for block, attn in zip(blocks, attns):
                h = attn(block(torch.cat([h, skips.pop()], dim=1), temb))
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))


if __name__ == "__main__":
    model = UNet()
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 1000, (2,))
    print("params:", sum(p.numel() for p in model.parameters()))
    print("output:", model(x, t).shape)
