"""A two-slot compositional digit generator — the toy 'text-to-image' model
that a GenEval-style harness probes.

Real GenEval measures whether a generator can honour *compositional* prompts:
the right number of the right objects, correctly bound. A single-digit MNIST
model can't fail compositionally, so we build the smallest model that can: it
paints a 28x56 canvas with up to two digits, conditioned on an UNORDERED pair
of requested classes (0-9, or 10 = 'empty').

Crucially the conditioning is order-agnostic — `emb[a] + emb[b]` — so the model
must decide *where* to put each digit and *how many* to draw on its own, exactly
the freedom that makes counting and object-binding hard for real T2I models.
"""

import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
from unet import UNet  # noqa: E402

EMPTY = 10  # the 'no object in this slot' token


class TwoSlotUNet(UNet):
    """U-Net conditioned on an unordered pair of class tokens. The two class
    embeddings are summed (so {a,b} and {b,a} are identical) and added to the
    time embedding, riding the same AdaGN pathway every ResBlock already has."""

    def __init__(self, num_tokens: int = 11, **unet_kwargs):
        # No spatial self-attention gated on a square 'res' — the canvas is
        # rectangular. The middle block still attends, which is shape-agnostic.
        unet_kwargs.setdefault("attn_resolutions", ())
        super().__init__(**unet_kwargs)
        self.class_emb = nn.Embedding(num_tokens, self.temb_dim)

    def forward(self, x, t, pair):
        # pair: (B, 2) long tokens. Order-agnostic sum.
        cond = self.time_embedding(t) + self.class_emb(pair).sum(dim=1)
        return self.forward_with_temb(x, cond)


if __name__ == "__main__":
    m = TwoSlotUNet(in_ch=1, image_size=28)
    x = torch.randn(2, 1, 28, 56)
    t = torch.randint(0, 200, (2,))
    pair = torch.tensor([[3, 8], [5, EMPTY]])
    print("params:", sum(p.numel() for p in m.parameters()))
    print("out:", m(x, t, pair).shape)
