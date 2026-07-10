"""LoRA / QLoRA: fine-tune by training a few small matrices, not the whole model.

Full fine-tuning updates every weight. LoRA freezes the base and adds a small
low-rank update `A @ B` to each linear — so you train a tiny fraction of the
parameters and keep one frozen copy of the base. QLoRA goes further: it also
*quantizes* the frozen base (here we emulate int8) so it costs less memory, while
the LoRA adapters stay full precision. We compare all three on accuracy and on how
many parameters each actually trains.

    python lora.py       # ~5 min on CPU

Reuses the shared task (sft_lib) and the GPT skeleton from project 08.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "28-sft-a-1b-base-model"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import sft_lib as L   # noqa: E402

OUT = HERE / "outputs"
STEPS = 700
RANK = 8


class LoRALinear(nn.Module):
    """Frozen base linear + trainable low-rank update, optionally int8-quantized base."""
    def __init__(self, base: nn.Linear, r=RANK, alpha=16, quantize=False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.quantize = quantize
        if quantize:                                       # emulate a frozen int8 base weight
            w = self.base.weight.data
            scale = w.abs().amax() / 127.0
            self.register_buffer("qw", (w / scale).round().clamp(-127, 127))
            self.qscale = float(scale)
        self.A = nn.Parameter(torch.zeros(base.out_features, r))
        self.B = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.scale = alpha / r

    def forward(self, x):
        if self.quantize:
            base_out = F.linear(x, self.qw * self.qscale)
        else:
            base_out = self.base(x)
        return base_out + (x @ self.B.t() @ self.A.t()) * self.scale


def inject_lora(model, quantize=False):
    targets = [(p, n, c) for p in model.modules()
               for n, c in p.named_children() if isinstance(c, nn.Linear)]
    for parent, name, child in targets:                    # collect BEFORE mutating the tree
        setattr(parent, name, LoRALinear(child, quantize=quantize))
    return model


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_base():
    """A lightly pretrained base (knows the format, ~0% correct) — the thing we adapt."""
    m = L.new_model(seed=0)
    L.train_sft(m, steps=150, corrupt=True, full_loss=True, seed=0)
    return m


def finetune(model, lr, steps=STEPS, seed=1):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = random.Random(seed)
    for step in range(steps):
        x, y, mask = L.sft_batch(64, rng)
        logits, _ = model(x)
        loss = L.masked_ce(logits, y, mask)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    # full fine-tune
    full = make_base()
    total = sum(p.numel() for p in full.parameters())
    finetune(full, lr=2e-3)
    results["full fine-tune"] = (L.accuracy(full), total)

    # LoRA (frozen base + rank-8 adapters)
    lora = inject_lora(make_base(), quantize=False)
    tp_lora = trainable_params(lora)
    finetune(lora, lr=5e-3)
    results["LoRA (r=8)"] = (L.accuracy(lora), tp_lora)

    # QLoRA (int8-quantized frozen base + adapters)
    qlora = inject_lora(make_base(), quantize=True)
    tp_qlora = trainable_params(qlora)
    finetune(qlora, lr=5e-3)
    results["QLoRA (int8 base)"] = (L.accuracy(qlora), tp_qlora)

    for name, (acc, tp) in results.items():
        print(f"{name:20s} acc {acc:.3f} | trainable {tp:,} ({100*tp/total:.1f}% of full)")

    plot(results, total)
    with open(OUT / "results.csv", "w") as f:
        f.write("method,accuracy,trainable_params,pct_of_full\n")
        for name, (acc, tp) in results.items():
            f.write(f"{name},{acc:.3f},{tp},{100*tp/total:.1f}\n")
    print(f"wrote {OUT/'lora.png'} + results.csv")


def plot(results, total):
    import plot_style as ps
    import matplotlib.pyplot as plt
    names = list(results)
    accs = [results[n][0] for n in names]
    pcts = [100 * results[n][1] / total for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4), dpi=110)
    for ax in (ax1, ax2):
        ax.set_facecolor(ps.SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(ps.BASELINE)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=8.5)
        ax.grid(True, axis="y", color=ps.GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.patch.set_facecolor(ps.SURFACE)
    cols = [ps.SERIES[0], ps.SERIES[1], ps.SERIES[3]]
    b1 = ax1.bar(range(len(names)), accs, color=cols, width=0.6)
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9, color=ps.INK_SECONDARY)
    ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, fontsize=8, rotation=12)
    ax1.set_ylim(0, 1); ax1.set_title("Comparable accuracy", color=ps.INK, fontsize=11,
                                       loc="left", pad=10)
    ax1.set_ylabel("exact-answer accuracy", color=ps.INK_SECONDARY, fontsize=10)
    b2 = ax2.bar(range(len(names)), pcts, color=cols, width=0.6)
    for i, v in enumerate(pcts):
        ax2.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9, color=ps.INK_SECONDARY)
    ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, fontsize=8, rotation=12)
    ax2.set_title("...at a fraction of the trainable params", color=ps.INK, fontsize=11,
                  loc="left", pad=10)
    ax2.set_ylabel("trainable params (% of full)", color=ps.INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "lora.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
