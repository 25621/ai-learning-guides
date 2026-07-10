"""BF16 vs FP8: fewer bits per number, until the math stops being stable.

Low precision is how modern accelerators go fast, but each format leaves less
headroom before rounding error corrupts a run. This CPU can't do real FP8 (that
needs an H100 + Transformer Engine), so we *emulate* the number formats with
fake-quantization (round every weight and activation onto the target grid, with a
straight-through gradient) and train the same small model four ways:

  * fp32        — full precision baseline
  * bf16        — real CPU autocast; 8 exponent bits like fp32, 7 mantissa bits
  * fp8 (e4m3)  — emulated; 4 exponent bits, 3 mantissa bits, no scaling
  * fp8 + scale — emulated e4m3 with per-tensor scaling into the representable range

The honest result at this scale: all four converge together — small models are
forgiving, so precision barely dents the loss curve. The difference the formats
*really* have shows up in a second experiment on numeric headroom: fp8's
representable range and precision are far narrower than bf16's, which is why real
fp8 training needs per-tensor scaling and higher-precision master weights, and why
it "quietly breaks" only once tensors span a wide dynamic range at scale.

    python precision_ablation.py       # ~4 min on CPU

Reuses the GPT skeleton from project 08.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT, CharData, cosine_lr, estimate_loss   # noqa: E402

OUT = HERE / "outputs"
FP8_MAX = 448.0                      # e4m3 max magnitude


def quant_e4m3(x):
    """Round onto the e4m3 grid: 3 mantissa bits, min-normal exponent -6, max 448."""
    xc = x.clamp(-FP8_MAX, FP8_MAX)
    a = xc.abs().clamp(min=2 ** -9)
    e = torch.floor(torch.log2(a)).clamp(min=-6)         # binade exponent
    step = 2.0 ** (e - 3)                                # mantissa LSB (3 bits)
    return torch.round(xc / step) * step


def fake_quant(x, scaled):
    if scaled:                                           # per-tensor scale into fp8 range
        s = FP8_MAX / x.abs().amax().clamp(min=1e-9)
        q = quant_e4m3(x * s) / s
    else:
        q = quant_e4m3(x)
    return x + (q - x).detach()                          # straight-through estimator


def bf16_roundtrip(x):
    q = x.to(torch.bfloat16).float()
    return x + (q - x).detach()                          # STE, at fp32 compute speed


class QLinear(nn.Module):
    """A bias-free Linear that fake-quantizes weights + inputs to the active format.

    We *emulate* every format (round to its grid with a straight-through gradient)
    rather than use real low-precision matmuls — CPU bf16 matmul is emulated and
    painfully slow, and there is no CPU fp8 at all. Emulation runs at fp32 speed and
    isolates exactly the thing we care about: rounding, not kernel performance.
    """
    mode = "off"                                         # off | bf16 | fp8 | fp8_scaled

    def __init__(self, lin):
        super().__init__()
        self.weight = lin.weight                         # share the Parameter (weight tying safe)

    def forward(self, x):
        w = self.weight
        if QLinear.mode == "bf16":
            w = bf16_roundtrip(w); x = bf16_roundtrip(x)
        elif QLinear.mode in ("fp8", "fp8_scaled"):
            sc = QLinear.mode == "fp8_scaled"
            w = fake_quant(w, sc); x = fake_quant(x, sc)
        return F.linear(x, w)


def wrap_linears(model):
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, QLinear(child))
    return model


def train(mode_label, qmode, data, steps=250):
    torch.manual_seed(0); torch.set_num_threads(12)
    cfg = Config(vocab_size=data.vocab_size, n_layer=4, n_head=4, n_embd=128, block_size=128)
    model = wrap_linears(GPT(cfg))
    QLinear.mode = qmode
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    curve = []
    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, 3e-3, warmup=40)
        if step % 25 == 0 or step == steps:
            QLinear.mode = "off"                         # eval master weights in fp32
            curve.append((step, estimate_loss(model, data, 32, iters=10)["val"]))
            QLinear.mode = qmode
        if step == steps:
            break
        x, y = data.batch("train", 32)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    QLinear.mode = "off"
    print(f"[{mode_label}] final val {curve[-1][1]:.3f}")
    return np.array(curve)


def headroom():
    """Relative round-trip error vs magnitude — the 'headroom' each format has.

    A small tensor (all ~1e-4) is crushed by fp8 unless per-tensor scaling lifts it
    into fp8's representable range first; bf16 keeps fp32's huge range so it never
    has this problem.
    """
    mags = np.logspace(-8, 3, 240)
    x = torch.tensor(mags, dtype=torch.float32)
    err_bf = ((x.to(torch.bfloat16).float() - x).abs() / x).numpy()
    err_f8 = ((quant_e4m3(x) - x).abs() / x.clamp(min=1e-30)).numpy()
    # concrete per-tensor-scaling demo on a tiny-magnitude tensor
    small = torch.full((4096,), 3e-4) * (1 + 0.1 * torch.randn(4096))
    e_noscale = ((quant_e4m3(small) - small).abs() / small.abs()).mean().item()
    s = FP8_MAX / small.abs().amax()
    e_scaled = ((quant_e4m3(small * s) / s - small).abs() / small.abs()).mean().item()
    return mags, err_bf, err_f8, e_noscale, e_scaled


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    text = Path(HERE / "data/corpus.txt").read_text(encoding="utf-8")
    data = CharData(text, block_size=128)
    runs = [
        ("fp32", "off"),
        ("bf16", "bf16"),
        ("fp8 (no scale)", "fp8"),
        ("fp8 + per-tensor scale", "fp8_scaled"),
    ]
    curves = {label: train(label, qmode, data) for label, qmode in runs}
    mags, err_bf, err_f8, e_noscale, e_scaled = headroom()
    print(f"tiny-tensor (~3e-4) mean relative error: fp8 no-scale {e_noscale:.3f} | "
          f"fp8 + per-tensor scale {e_scaled:.3f}")
    plot(curves, mags, err_bf, err_f8)
    with open(OUT / "results.csv", "w") as f:
        f.write("format,final_val_loss\n")
        for label in curves:
            f.write(f"{label},{curves[label][-1,1]:.3f}\n")
        f.write("\nprecision_probe,mean_relative_error\n")
        f.write(f"fp8_no_scale_tiny_tensor,{e_noscale:.3f}\n")
        f.write(f"fp8_scaled_tiny_tensor,{e_scaled:.3f}\n")
    print(f"wrote {OUT/'precision.png'} + results.csv")


def plot(curves, mags, err_bf, err_f8):
    import plot_style as ps
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4), dpi=110)
    for ax in (ax1, ax2):
        ax.set_facecolor(ps.SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(ps.BASELINE)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=9)
        ax.grid(True, color=ps.GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.patch.set_facecolor(ps.SURFACE)

    # left: training curves — at toy scale they overlap (honest)
    order = ["fp32", "bf16", "fp8 (no scale)", "fp8 + per-tensor scale"]
    colors = {"fp32": ps.INK, "bf16": ps.SERIES[1],
              "fp8 (no scale)": ps.SERIES[2], "fp8 + per-tensor scale": ps.SERIES[3]}
    for label in order:
        c = curves[label]
        ax1.plot(c[:, 0], c[:, 1], color=colors[label], label=label,
                 lw=2.4 if label == "fp32" else 1.5, ls="--" if label == "fp32" else "-")
    ax1.legend(frameon=False, fontsize=8, title="format")
    ax1.set_title("At toy scale, all four converge together", color=ps.INK, fontsize=11,
                  loc="left", pad=10)
    ax1.set_xlabel("training step", color=ps.INK_SECONDARY, fontsize=10)
    ax1.set_ylabel("validation loss (nats/char)", color=ps.INK_SECONDARY, fontsize=10)

    # right: precision headroom — where the formats really differ
    ax2.loglog(mags, np.clip(err_bf, 1e-4, 1), color=ps.SERIES[1], label="bf16")
    ax2.loglog(mags, np.clip(err_f8, 1e-4, 1), color=ps.SERIES[2], label="fp8 e4m3")
    ax2.axvspan(2 ** -9, 448, color=ps.SERIES[1], alpha=0.06)
    ax2.text(2 ** -8, 0.5, "fp8 usable range", color=ps.INK_MUTED, fontsize=7.5, rotation=90)
    ax2.legend(frameon=False, fontsize=8, title="round-trip error")
    ax2.set_title("...but fp8's headroom is far narrower", color=ps.INK, fontsize=11,
                  loc="left", pad=10)
    ax2.set_xlabel("value magnitude", color=ps.INK_SECONDARY, fontsize=10)
    ax2.set_ylabel("relative rounding error", color=ps.INK_SECONDARY, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "precision.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
