"""The InstructPix2Pix editor and the synthetic-data operators.

Two pieces:

1. `INSTRUCTIONS` + `apply_edit` — the synthetic dataset. Real InstructPix2Pix
   builds its (original, instruction, edited) triples with a large language
   model (to invent the instruction and a before/after caption pair) and
   Prompt-to-Prompt (to render a matched image pair that differs only in the
   edited detail). We have neither offline, so we substitute a handful of
   *procedural* edits: each instruction is a deterministic image op. The data
   pipeline is the point — an editor is trained on generated pairs, so no human
   ever hand-edits an image — and it is identical whether the pairs come from
   GPT+P2P or from these ops.

2. `InstructEditor` — a U-Net that conditions on BOTH the instruction (an
   embedding, like a class label) and the original image (concatenated to the
   noisy input, the InstructPix2Pix trick). At inference it edits in a single
   denoising loop: no per-image optimization the way older editing methods need.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "24-ddpm-on-mnist"))
from unet import UNet  # noqa: E402

INSTRUCTIONS = ["thicken", "thin", "invert", "flip", "shift right"]


def apply_edit(x, instr_id):
    """Deterministic edit op on a [-1,1] batch (stands in for GPT+P2P output)."""
    name = INSTRUCTIONS[instr_id]
    if name == "thicken":
        p = (x + 1) / 2
        return F.max_pool2d(p, 3, 1, 1) * 2 - 1
    if name == "thin":
        p = (x + 1) / 2
        return (-F.max_pool2d(-p, 3, 1, 1)) * 2 - 1
    if name == "invert":
        return -x
    if name == "flip":
        return torch.flip(x, dims=[-2])          # upside-down
    if name == "shift right":
        return torch.roll(x, shifts=4, dims=-1)
    raise ValueError(name)


class InstructEditor(UNet):
    def __init__(self, **unet_kwargs):
        super().__init__(in_ch=1, **unet_kwargs)  # out_conv stays 1-channel
        base_ch = self.in_conv.out_channels
        # input is [noisy_edited ; original] -> 2 channels
        self.in_conv = nn.Conv2d(2, base_ch, 3, padding=1)
        self.instr_emb = nn.Embedding(len(INSTRUCTIONS), self.temb_dim)

    def forward(self, x, t, original, instr):
        cond = self.time_embedding(t) + self.instr_emb(instr)
        return self.forward_with_temb(torch.cat([x, original], dim=1), cond)


if __name__ == "__main__":
    m = InstructEditor()
    x = torch.randn(2, 1, 28, 28)
    out = m(x, torch.randint(0, 200, (2,)), x, torch.tensor([0, 1]))
    print("output", out.shape, "| params", sum(p.numel() for p in m.parameters()))
