"""FP8 serving: weights, activations and the KV cache at 8 bits — and why scaling is not optional.

TransformerEngine needs a Hopper GPU, and this box has neither. So we do what the
hardware would do, in software: round every number to the nearest value the FP8
E4M3 (or E5M2) format can represent, and run the model on those. The arithmetic
the model sees is the arithmetic FP8 tensor cores would give it, so *quality* is
measured exactly. Throughput is the one thing we cannot measure — no FP8 units —
so we compute it from the bandwidth model the guide gives, and say so plainly.

The FP8 formats:

  E4M3   4 exponent bits, 3 mantissa bits. Range +-448, ~2 decimal digits.
         The format for weights and activations.
  E5M2   5 exponent bits, 2 mantissa bits. Wider range, coarser steps.
         The format for gradients (and, here, a cautionary tale).

The headline experiment is the *scale factor*. FP8's smallest normal magnitude is
about 2e-3. Real activations routinely live below that. Without a per-tensor scale
they simply become zero, and the model breaks — which is why every FP8 recipe in
production is really an FP8-plus-scaling recipe.

Run:  python3 fp8.py           (~8 min)
      python3 fp8.py --plot    (redraw from outputs/results.json)
"""

import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "59-quantize-a-7b-model"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))

import quant_lib as Q  # noqa: E402
from quant_lib import E  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoints")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PPL_SEQ, PPL_LEN = 16, 512
MMLU_PER_SUBJECT = 3
torch.set_num_threads(12)

# (exponent bits, mantissa bits, max finite value)
FORMATS = {"e4m3": (4, 3, 448.0), "e5m2": (5, 2, 57344.0)}


# ----------------------------------------------------------------- the format
def fp8_quantize(x, fmt="e4m3", scale=None):
    """Round x to the nearest FP8 value. `scale` is the per-tensor amax scaling.

    An FP8 number is sign x 2^e x 1.mmm. Rounding to it means: find the exponent,
    then round the mantissa to `m` bits. Values below the smallest normal
    (2^(1-bias)) fall into the subnormal range and, without a scale, flush to zero.
    """
    e_bits, m_bits, fmax = FORMATS[fmt]
    bias = 2 ** (e_bits - 1) - 1

    if scale is not None:
        x = x / scale

    sign = torch.sign(x)
    a = x.abs().clamp(max=fmax)

    e = torch.floor(torch.log2(a.clamp(min=1e-30)))          # which binade the value is in
    e = e.clamp(min=1 - bias)                                # subnormals all share the lowest
    step = torch.pow(2.0, e - m_bits)                        # spacing of the mantissa grid
    # Rounding to the grid is also what implements underflow: anything smaller than
    # half the smallest subnormal step (2^-9 for E4M3) simply rounds to zero.
    out = sign * torch.round(a / step) * step

    return out * scale if scale is not None else out


def min_representable(fmt="e4m3"):
    e_bits, m_bits, _ = FORMATS[fmt]
    bias = 2 ** (e_bits - 1) - 1
    return 2.0 ** (1 - bias - m_bits)                        # smallest subnormal


def amax_scale(x, fmt="e4m3"):
    """The scale every production FP8 kernel keeps: map the tensor's max onto the format's max."""
    fmax = FORMATS[fmt][2]
    amax = x.abs().max().clamp(min=1e-12)
    return amax / fmax


# -------------------------------------------------------- patching the model
class FP8Linear(nn.Module):
    """A linear layer that quantizes its weights, and optionally its inputs, to FP8."""

    def __init__(self, lin, fmt="e4m3", quant_act=False, scaled=True):
        super().__init__()
        self.lin = lin
        self.quant_act, self.fmt, self.scaled = quant_act, fmt, scaled
        w = lin.weight.data.float()
        s = amax_scale(w, fmt) if scaled else None
        lin.weight.data = fp8_quantize(w, fmt, s)            # weights: quantized once, kept

    def forward(self, x):
        if self.quant_act:
            # Activations are quantized *dynamically*: their range moves with the input,
            # so the scale is recomputed per tensor, exactly as TransformerEngine does.
            s = amax_scale(x, self.fmt) if self.scaled else None
            x = fp8_quantize(x.float(), self.fmt, s).to(x.dtype)
        return self.lin(x)


def patch_fp8(model, fmt="e4m3", acts=False, scaled=True):
    """Replace every linear inside the transformer blocks with its FP8 twin."""
    n = 0
    for blk in model.model.layers:
        for parent in [blk.self_attn, blk.mlp]:
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, FP8Linear(child, fmt, acts, scaled))
                    n += 1
    return n


def patch_kv_fp8(model, fmt="e4m3"):
    """Round K and V to FP8 right where the cache would store them.

    We hook the k_proj / v_proj outputs: whatever they produce is what the cache
    holds, so quantizing here is equivalent to an FP8 KV cache (and it also covers
    the current token, which a real kernel would too).
    """
    handles = []
    for blk in model.model.layers:
        for proj in (blk.self_attn.k_proj, blk.self_attn.v_proj):
            handles.append(proj.register_forward_hook(
                lambda m, i, o, f=fmt: fp8_quantize(o.float(), f,
                                                    amax_scale(o.float(), f)).to(o.dtype)))
    return handles


# ------------------------------------------------------------------ the study
def kv_bytes_per_token(layers=32, kv_heads=8, d_head=128, bytes_per=2):
    return 2 * layers * kv_heads * d_head * bytes_per


def evaluate(label, patch, text, mmlu_items):
    wrapper = E.Model(MODEL)
    model, tok = wrapper.model, wrapper.tok
    info = patch(model) if patch else ""
    ppl = Q.perplexity(model, tok, text, seqlen=PPL_LEN, n_seq=N_PPL_SEQ)
    acc, correct, n = Q.mmlu_accuracy(wrapper, mmlu_items)
    row = dict(label=label, ppl=ppl, mmlu=acc)
    print(f"  {label:34s} | ppl {ppl:9.3f} | MMLU {acc:.3f}", flush=True)
    del wrapper, model
    return row


def underflow_probe():
    """Why the scale exists, in four numbers."""
    rows = []
    for name, mag in [("a typical activation (~1.0)", 1.0),
                      ("a small activation (~3e-4)", 3e-4)]:
        x = torch.full((256,), mag) * (1 + 0.1 * torch.randn(256))
        for scaled in (False, True):
            s = amax_scale(x, "e4m3") if scaled else None
            q = fp8_quantize(x, "e4m3", s)
            err = ((q - x).abs() / x.abs()).mean().item()
            rows.append(dict(case=name, scaled=scaled, rel_err=err,
                             zeroed=float((q == 0).float().mean())))
            print(f"  {name:28s} | scale {'on ' if scaled else 'off'} | "
                  f"mean relative error {err:7.2%} | flushed to zero {rows[-1]['zeroed']:5.0%}")
    return rows


def plot(rows, probe):
    # 1. quality across the recipes — shown as the *cost* over bf16, because the
    #    absolute perplexities differ by a few percent and a zero-based bar chart
    #    would render them all as the same bar.
    base = rows[0]["ppl"]
    body = rows[1:]
    fig, ax = ps.new_axes(7.6, 4.4)
    colors = ([ps.SERIES[0], ps.SERIES[0], ps.SERIES[0], ps.SERIES[1],
               ps.SERIES[2], ps.SERIES[2]] + [ps.SERIES[2]] * 4)[:len(body)]
    deltas = [(r["ppl"] / base - 1) * 100 for r in body]
    ypos = list(range(len(body)))[::-1]                     # first recipe at the top
    ax.barh(ypos, deltas, color=colors, height=0.62, zorder=3)
    for y, d, r in zip(ypos, deltas, body):
        ax.annotate(f"+{d:.1f}%   (ppl {r['ppl']:.2f})", xy=(d, y), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    color=ps.INK_SECONDARY, fontsize=9)
    ax.axvline(0, color=ps.INK_SECONDARY, lw=1.2)
    ax.annotate(f"bf16 baseline (ppl {base:.2f})", xy=(0, len(body) - 0.4),
                xytext=(4, 0), textcoords="offset points",
                color=ps.INK_SECONDARY, fontsize=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["label"] for r in body], fontsize=9)
    ax.set_xlim(0, max(deltas) * 1.65)
    ax.grid(axis="y", visible=False)
    ps.finish(fig, ax, "What each FP8 recipe costs in quality",
              "perplexity increase over bf16 (%)", "",
              os.path.join(OUT, "quality.png"))

    # 2. the underflow probe — why scaling is mandatory
    fig, ax = ps.new_axes(7.2, 4.0)
    cases = [p for p in probe if "small" in p["case"]]
    others = [p for p in probe if "typical" in p["case"]]
    xs = ["typical values\nno scale", "typical values\nscaled",
          "small values\nno scale", "small values\nscaled"]
    ys = [others[0]["rel_err"], others[1]["rel_err"],
          cases[0]["rel_err"], cases[1]["rel_err"]]
    ax.bar(xs, ys, color=[ps.SERIES[1], ps.SERIES[0], ps.SERIES[2], ps.SERIES[0]],
           width=0.6, zorder=3)
    for i, v in enumerate(ys):
        ax.annotate(f"{v:.1%}", xy=(i, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=ps.INK_SECONDARY, fontsize=9)
    ax.annotate("everything underflows to zero", xy=(2, ys[2]), xytext=(0, -30),
                textcoords="offset points", ha="center", color=ps.SERIES[2], fontsize=9)
    ps.finish(fig, ax, "FP8 without a scale factor is not FP8, it is zero",
              "", "mean relative error", os.path.join(OUT, "underflow.png"))

    # 3. memory / bandwidth: what FP8 actually buys on real hardware
    fig, ax = ps.new_axes(7.2, 4.2)
    N = 8.03e9                                             # Llama-3-8B
    fmts = ["bf16", "fp8"]
    w = [2 * N / 1e9, 1 * N / 1e9]
    kv = [kv_bytes_per_token(bytes_per=2) * 8192 * 32 / 1e9,
          kv_bytes_per_token(bytes_per=1) * 8192 * 32 / 1e9]
    x = range(2)
    ax.bar(x, w, 0.5, color=ps.SERIES[0], label="weights", zorder=3)
    ax.bar(x, kv, 0.5, bottom=w, color=ps.SERIES[1], label="KV cache (8k ctx, batch 32)",
           zorder=3)
    for i in x:
        ax.annotate(f"{w[i]+kv[i]:.0f} GB", xy=(i, w[i] + kv[i]), xytext=(0, 5),
                    textcoords="offset points", ha="center",
                    color=ps.INK_SECONDARY, fontsize=10)
    ax.set_xticks(list(x)); ax.set_xticklabels(fmts)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=9)
    ax.annotate("decode is bandwidth-bound:\nhalf the bytes ~ twice the tokens/s",
                xy=(1, w[1] + kv[1]), xytext=(-10, 46), textcoords="offset points",
                ha="right", color=ps.INK_SECONDARY, fontsize=9)
    ps.finish(fig, ax, "Llama-3-8B on one GPU: what FP8 halves",
              "", "GB of HBM read per decode step", os.path.join(OUT, "memory.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
    cache = os.path.join(CKPT, "results.json")

    if args.plot:
        d = json.load(open(cache))
        plot(d["rows"], d["probe"])
        return

    text = Q.wikitext()
    mmlu_items = E.load_mmlu(per_subject=MMLU_PER_SUBJECT, seed=0)

    print("[1/2] quality, recipe by recipe")
    rows = [
        evaluate("bf16 baseline", None, text, mmlu_items),
        evaluate("E4M3 weights", lambda m: patch_fp8(m, "e4m3"), text, mmlu_items),
        evaluate("E4M3 weights + activations",
                 lambda m: patch_fp8(m, "e4m3", acts=True), text, mmlu_items),
        evaluate("E4M3 weights + acts + KV cache",
                 lambda m: (patch_fp8(m, "e4m3", acts=True), patch_kv_fp8(m, "e4m3")),
                 text, mmlu_items),
        evaluate("E5M2 weights (fewer mantissa bits)",
                 lambda m: patch_fp8(m, "e5m2"), text, mmlu_items),
        evaluate("E4M3 weights, NO scale",
                 lambda m: patch_fp8(m, "e4m3", scaled=False), text, mmlu_items),
        evaluate("E4M3 weights + acts, NO scale",
                 lambda m: patch_fp8(m, "e4m3", acts=True, scaled=False),
                 text, mmlu_items),
    ]

    print("\n[2/2] why the scale factor exists")
    probe = underflow_probe()

    json.dump(dict(rows=rows, probe=probe), open(cache, "w"), indent=1)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "ppl", "mmlu"])
        w.writeheader(); w.writerows(rows)
    plot(rows, probe)


if __name__ == "__main__":
    main()
