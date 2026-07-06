"""LoRA (Low-Rank Adaptation) for the phase-5 U-Net.

The whole idea in one class: leave a pretrained Conv2d's weights frozen and
add a trainable low-rank *residual* alongside it,

    y = W x  +  (alpha / r) * (B (A x))

where A projects the input down to a tiny rank-r space and B projects back up.
A starts random, B starts at ZERO, so the residual is exactly zero at init and
the wrapped layer behaves identically to the frozen original on step one — you
are only ever learning a small correction. For a Conv2d, `A` is a conv with the
base kernel/stride/padding but only `r` output channels, and `B` is a 1x1 conv
back to the base's output channels. The number of new parameters is
r*(in + out) per layer instead of in*out*k*k — a few percent of the model.

`inject_lora` walks a model and swaps its Conv2d layers for wrapped versions in
place, then returns the list of adapters so a trainer can optimize only those.
"""

from torch import nn


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int = 4, alpha: float | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)  # the pretrained weights never move
        self.scale = (alpha if alpha is not None else rank) / rank
        self.down = nn.Conv2d(
            base.in_channels, rank, base.kernel_size,
            base.stride, base.padding, bias=False,
        )
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)  # residual == 0 at init

    def forward(self, x):
        return self.base(x) + self.scale * self.up(self.down(x))


def inject_lora(model, rank: int = 4, alpha: float | None = None,
                skip=("in_conv", "out_conv")):
    """Replace every Conv2d in `model` (except the input/output stems) with a
    LoRA-wrapped copy. Freezes all original parameters. Returns the adapters."""
    for p in model.parameters():
        p.requires_grad_(False)
    # Collect targets BEFORE mutating — replacing modules mid-traversal would
    # make the walker descend into the freshly wrapped layers and recurse.
    targets = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Conv2d) and not any(full.endswith(s) for s in skip):
                targets.append((parent, child_name, child))
    adapters = []
    for parent, child_name, child in targets:
        lora = LoRAConv2d(child, rank, alpha)
        setattr(parent, child_name, lora)
        adapters.append(lora)
    return adapters


def lora_parameters(adapters):
    for a in adapters:
        yield from a.down.parameters()
        yield from a.up.parameters()


def lora_state_dict(model):
    """Only the LoRA tensors — this is the few-megabyte file you would share."""
    return {k: v for k, v in model.state_dict().items()
            if ".down." in k or ".up." in k}
