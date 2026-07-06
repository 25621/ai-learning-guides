"""ControlNet, built from scratch on top of the phase-5 U-Net.

The recipe (Zhang et al. 2023):
  1. Freeze the pretrained base U-Net.
  2. Make a trainable COPY of its encoder (the down blocks + middle) — the
     'control branch'. Initialize it from the base's own weights.
  3. Feed the conditioning image (here a Canny-style edge map) into the branch
     through a zero-convolution, and add the branch's features back into the
     base's skip connections, each through its own zero-convolution.

Zero-convolutions (1x1 convs initialized to all-zero weight and bias) are the
whole trick: at step 0 every injection is exactly zero, so the controlled model
reproduces the frozen base bit-for-bit and training starts from a known-good
point. The branch then learns how much control to add without ever destabilizing
the pretrained weights.

We re-implement the base U-Net's forward here so we can splice control features
into the skip list — the base's own `forward` hides that list internally.
"""

import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "24-ddpm-on-mnist"))
from unet import UNet  # noqa: E402


class ZeroConv(nn.Conv2d):
    def __init__(self, in_c, out_c):
        super().__init__(in_c, out_c, kernel_size=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


def edge_map(x):
    """Canny-style edge map: Sobel gradient magnitude of a [-1,1] image,
    normalized per-image to [-1,1]. A cheap, dependency-free stand-in for Canny
    that gives the control branch a clean structural signal."""
    p = (x + 1) / 2
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=p.dtype)
    ky = kx.t()
    k = torch.stack([kx, ky]).unsqueeze(1)  # (2,1,3,3)
    g = F.conv2d(p, k, padding=1)
    mag = (g[:, :1] ** 2 + g[:, 1:] ** 2).sqrt()
    mag = mag / (mag.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
    return mag * 2 - 1


class ControlledUNet(nn.Module):
    def __init__(self, base: UNet):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        # trainable copy of the encoder + middle
        self.c_in_conv = copy.deepcopy(base.in_conv)
        self.c_down_blocks = copy.deepcopy(base.down_blocks)
        self.c_down_attns = copy.deepcopy(base.down_attns)
        self.c_downsamples = copy.deepcopy(base.downsamples)
        self.c_mid_block1 = copy.deepcopy(base.mid_block1)
        self.c_mid_attn = copy.deepcopy(base.mid_attn)
        self.c_mid_block2 = copy.deepcopy(base.mid_block2)

        base_ch = base.in_conv.out_channels
        # hint enters the branch through a small encoder ending in a zero-conv
        self.hint = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.SiLU(),
        )
        self.hint_zero = ZeroConv(base_ch, base_ch)
        # one zero-conv per skip the encoder produces, plus one for the middle
        self.zero_convs = nn.ModuleList(ZeroConv(c, c) for c in base.skip_chs)
        mid_ch = self._mid_ch(base)
        self.mid_zero = ZeroConv(mid_ch, mid_ch)

    @staticmethod
    def _mid_ch(base):
        return base.mid_block1.conv2.out_channels

    def _control_encode(self, x, hint, temb):
        """Run the trainable branch; return (skips, mid) mirroring the base."""
        h = self.c_in_conv(x) + self.hint_zero(self.hint(hint))
        skips = [h]
        for level, (blocks, attns) in enumerate(zip(self.c_down_blocks, self.c_down_attns)):
            for block, attn in zip(blocks, attns):
                h = attn(block(h, temb))
                skips.append(h)
            if not isinstance(self.c_downsamples[level], nn.Identity):
                h = self.c_downsamples[level](h)
                skips.append(h)
        mid = self.c_mid_block2(self.c_mid_attn(self.c_mid_block1(h, temb)), temb)
        return skips, mid

    def forward(self, x, t, hint):
        b = self.base
        temb = b.time_embedding(t)
        c_skips, c_mid = self._control_encode(x, hint, temb)

        # frozen base encoder, adding control features into every skip
        h = b.in_conv(x)
        skips = [h + self.zero_convs[0](c_skips[0])]
        si = 1
        for level, (blocks, attns) in enumerate(zip(b.down_blocks, b.down_attns)):
            for block, attn in zip(blocks, attns):
                h = attn(block(h, temb))
                skips.append(h + self.zero_convs[si](c_skips[si])); si += 1
            if not isinstance(b.downsamples[level], nn.Identity):
                h = b.downsamples[level](h)
                skips.append(h + self.zero_convs[si](c_skips[si])); si += 1

        h = b.mid_block2(b.mid_attn(b.mid_block1(h, temb)), temb)
        h = h + self.mid_zero(c_mid)

        for blocks, attns, up in zip(b.up_blocks, b.up_attns, b.upsamples):
            for block, attn in zip(blocks, attns):
                h = attn(block(torch.cat([h, skips.pop()], dim=1), temb))
            h = up(h)
        return b.out_conv(F.silu(b.out_norm(h)))

    def control_parameters(self):
        for n, p in self.named_parameters():
            if not n.startswith("base."):
                yield p


if __name__ == "__main__":
    base = UNet()
    m = ControlledUNet(base)
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 200, (2,))
    hint = edge_map(x)
    out = m(x, t, hint)
    print("output", out.shape)
    # zero-init check: at start the control branch contributes nothing.
    print("identity at init:", torch.allclose(out, base(x, t), atol=1e-6))
    print("control params:", sum(p.numel() for p in m.control_parameters()))
