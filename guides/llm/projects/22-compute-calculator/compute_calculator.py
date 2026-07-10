"""The 6ND compute calculator: predict a training run's FLOPs on the back of an
envelope, then check the prediction against a real, measured run.

A forward+backward pass over a dense transformer with N (non-embedding)
parameters on D tokens costs about 6ND FLOPs: ~2ND for the forward pass and ~4ND
for the backward. This script implements that formula and validates it three ways:

  1. against PyTorch's own FLOP counter (should agree to ~1%),
  2. by splitting measured FLOPs into forward vs backward (should be ~2:4),
  3. by turning measured FLOPs and wall-time into an achieved FLOP/s and an MFU
     (model FLOPs utilization) against this machine's measured matmul peak.

    python compute_calculator.py            # ~1 min on CPU

Reuses the GPT skeleton from project 08.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT   # noqa: E402

OUT = HERE / "outputs"


def flops_6nd(n_params, n_tokens):
    """The whole cost of a training step, on the back of an envelope."""
    return 6 * n_params * n_tokens


def measure_cpu_peak_flops(size=2048, reps=3):
    """This machine's realized dense-matmul throughput — our 'hardware peak'."""
    a = torch.randn(size, size); b = torch.randn(size, size)
    a @ b                                                    # warm up
    t0 = time.time()
    for _ in range(reps):
        a @ b
    dt = (time.time() - t0) / reps
    return (2 * size ** 3) / dt                              # matmul is 2*n^3 FLOPs


def main():
    torch.manual_seed(0); torch.set_num_threads(12)
    OUT.mkdir(parents=True, exist_ok=True)

    batch, block = 12, 256
    cfg = Config(vocab_size=65, n_layer=6, n_head=8, n_embd=384, block_size=block)
    model = GPT(cfg)
    N = model.num_params()
    D = batch * block
    x = torch.randint(0, 65, (batch, block)); y = torch.randint(0, 65, (batch, block))

    # ---- 1) measured FLOPs vs the 6ND estimate ----
    fc = FlopCounterMode(display=False)
    with fc:
        _, loss = model(x, y); loss.backward()
    measured_total = fc.get_total_flops()
    model.zero_grad(set_to_none=True)

    fc_fwd = FlopCounterMode(display=False)
    with fc_fwd, torch.no_grad():
        model(x, y)
    measured_fwd = fc_fwd.get_total_flops()
    measured_bwd = measured_total - measured_fwd
    est = flops_6nd(N, D)

    # ---- 2) wall-time -> achieved FLOP/s -> MFU ----
    for _ in range(2):                                       # warm up
        model.zero_grad(set_to_none=True)
        _, l = model(x, y); l.backward()
    reps = 8
    t0 = time.time()
    for _ in range(reps):
        model.zero_grad(set_to_none=True)
        _, l = model(x, y); l.backward()
    step_time = (time.time() - t0) / reps
    achieved = measured_total / step_time
    peak = measure_cpu_peak_flops()
    mfu = achieved / peak

    print(f"model: {N/1e6:.2f}M non-embedding params | batch {batch} x block {block} "
          f"= {D} tokens/step")
    print(f"6ND estimate      : {est/1e9:8.2f} GFLOP")
    print(f"measured total    : {measured_total/1e9:8.2f} GFLOP  (ratio {measured_total/est:.3f})")
    print(f"  forward         : {measured_fwd/1e9:8.2f} GFLOP  (~2ND -> {measured_fwd/(2*N*D):.2f}x)")
    print(f"  backward        : {measured_bwd/1e9:8.2f} GFLOP  (~4ND -> {measured_bwd/(4*N*D):.2f}x)")
    print(f"step time         : {step_time*1e3:8.1f} ms")
    print(f"achieved          : {achieved/1e9:8.1f} GFLOP/s")
    print(f"CPU matmul peak   : {peak/1e9:8.1f} GFLOP/s")
    print(f"MFU               : {mfu*100:8.1f} %")

    # ---- outputs ----
    with open(OUT / "results.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"N_params_millions,{N/1e6:.3f}\n")
        f.write(f"tokens_per_step,{D}\n")
        f.write(f"est_6ND_gflop,{est/1e9:.2f}\n")
        f.write(f"measured_total_gflop,{measured_total/1e9:.2f}\n")
        f.write(f"measured_fwd_gflop,{measured_fwd/1e9:.2f}\n")
        f.write(f"measured_bwd_gflop,{measured_bwd/1e9:.2f}\n")
        f.write(f"measured_over_estimate,{measured_total/est:.3f}\n")
        f.write(f"step_time_ms,{step_time*1e3:.1f}\n")
        f.write(f"achieved_gflops,{achieved/1e9:.1f}\n")
        f.write(f"cpu_peak_gflops,{peak/1e9:.1f}\n")
        f.write(f"mfu_percent,{mfu*100:.1f}\n")
    plot(est, measured_total, measured_fwd, measured_bwd, N, D)


def plot(est, total, fwd, bwd, N, D):
    import plot_style as ps
    fig, (ax1, ax2) = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(
        1, 2, figsize=(9.4, 4.2), dpi=110)
    for ax in (ax1, ax2):
        ax.set_facecolor(ps.SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(ps.BASELINE)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=9)
        ax.grid(True, axis="y", color=ps.GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.patch.set_facecolor(ps.SURFACE)

    # left: 6ND estimate vs measured total
    ax1.bar(["6ND\nestimate", "measured\n(PyTorch)"], [est / 1e9, total / 1e9],
            color=[ps.SERIES[0], ps.SERIES[1]], width=0.6)
    for i, v in enumerate([est / 1e9, total / 1e9]):
        ax1.text(i, v + total / 1e9 * 0.01, f"{v:.1f}", ha="center", fontsize=9,
                 color=ps.INK_SECONDARY)
    ax1.set_title("6ND predicts the real FLOP count", color=ps.INK, fontsize=11,
                  loc="left", pad=10)
    ax1.set_ylabel("GFLOP per step", color=ps.INK_SECONDARY, fontsize=10)

    # right: forward vs backward split (the 2:4 that makes the 6)
    ax2.bar(["forward\n(~2ND)", "backward\n(~4ND)"], [fwd / 1e9, bwd / 1e9],
            color=[ps.SERIES[3], ps.SERIES[2]], width=0.6)
    for i, (v, ref) in enumerate([(fwd / 1e9, 2 * N * D / 1e9), (bwd / 1e9, 4 * N * D / 1e9)]):
        ax2.text(i, v + bwd / 1e9 * 0.01, f"{v:.1f}", ha="center", fontsize=9,
                 color=ps.INK_SECONDARY)
    ax2.set_title("Backward costs ~2x forward: 6 = 2 + 4", color=ps.INK, fontsize=11,
                  loc="left", pad=10)
    ax2.set_ylabel("GFLOP per step", color=ps.INK_SECONDARY, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "flop_accounting.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT/'flop_accounting.png'} + results.csv")


if __name__ == "__main__":
    main()
