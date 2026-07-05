"""Class-conditional U-Net: the label rides the same AdaGN pathway as time.

Project 24's ResBlocks already modulate their GroupNorm with a (scale, shift)
computed from the conditioning vector — that is AdaGN (adaptive group norm,
also called FiLM). Making the model class-conditional therefore takes three
lines: learn an embedding per class, ADD it to the time embedding, and let
every ResBlock see the sum. No other architecture change is needed.
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "24-ddpm-on-mnist"))
from unet import UNet  # noqa: E402


class ConditionalUNet(UNet):
    def __init__(self, num_classes: int = 10, **unet_kwargs):
        super().__init__(**unet_kwargs)
        self.label_emb = nn.Embedding(num_classes, self.temb_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = self.time_embedding(t) + self.label_emb(y)
        return self.forward_with_temb(x, cond)


if __name__ == "__main__":
    model = ConditionalUNet()
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 1000, (2,))
    y = torch.tensor([3, 7])
    print("params:", sum(p.numel() for p in model.parameters()))
    print("output:", model(x, t, y).shape)
