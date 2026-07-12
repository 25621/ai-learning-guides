"""Quantize a real model four ways and see what each bit costs.

The guide asks for a 7B. A 7B does not fit in this CPU-only box, so we run the
same recipes on Qwen2.5-0.5B — a real, modern, GQA transformer with a 152k
vocabulary. Everything transfers except the absolute numbers, and the *ordering*
of the methods (which is the point) is the same one the papers report at 7B.
If anything this is the harder test: a 0.5B model has less redundancy than a 7B,
so 4-bit rounding hurts it more.

  fp16 (baseline)   16 bits/weight, no rounding
  INT8 RTN          8 bits, round-to-nearest, data-free
  INT4 RTN          4 bits, group-128, data-free      <- the naive 4-bit result
  INT4 GPTQ         4 bits, group-128, Hessian error feedback
  INT4 AWQ          4 bits, group-128, activation-aware scaling

Run:  python3 quantize.py           (~9 min)
      python3 quantize.py --plot    (redraw from outputs/results.csv)
"""

import argparse
import csv
import json
import os
import sys
import time

import torch

import quant_lib as Q
from quant_lib import E  # the Phase-8 eval stack

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-train-a-bpe-from-scratch"))
import plot_style as ps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoints")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# (label, method, bits, group_size)
RECIPES = [
    ("fp16",      None,   16, -1),
    ("INT8 RTN",  "rtn",   8, -1),      # per-channel: one scale per output row
    ("INT4 RTN",  "rtn",   4, 128),
    ("INT4 GPTQ", "gptq",  4, 128),
    ("INT4 AWQ",  "awq",   4, 128),
]

N_CALIB, CALIB_LEN = 16, 256            # calibration set for GPTQ/AWQ
N_PPL_SEQ, PPL_LEN = 16, 512            # wikitext-2 perplexity windows
MMLU_PER_SUBJECT = 3                    # 57 subjects x 3 = 171 questions


def calib_ids(tok, text):
    """A small slice of real text — GPTQ needs the input statistics, not the labels."""
    ids = tok(text, return_tensors="pt").input_ids[0]
    return torch.stack([ids[i * CALIB_LEN:(i + 1) * CALIB_LEN] for i in range(N_CALIB)])


def evaluate(label, method, bits, group, text, mmlu_items):
    """Load a fresh copy of the model, quantize it, and score it."""
    t0 = time.time()
    wrapper = E.Model(MODEL)                        # fresh fp32 weights every time
    model, tok = wrapper.model, wrapper.tok

    n_quant_time = 0.0
    if method is not None:
        tq = time.time()
        Q.quantize_model(model, method, bits, group,
                         calib_ids=calib_ids(tok, text) if method != "rtn" else None)
        n_quant_time = time.time() - tq

    mb, block_params, total = Q.model_bytes(model, bits, group)
    ppl = Q.perplexity(model, tok, text, seqlen=PPL_LEN, n_seq=N_PPL_SEQ)
    acc, correct, n = Q.mmlu_accuracy(wrapper, mmlu_items)
    lo, hi = E.wilson_ci(correct, n)[1:]

    row = dict(label=label, bits=bits, group=group, ppl=ppl, mmlu=acc,
               mmlu_lo=lo, mmlu_hi=hi, mb=mb / 1e6,
               eff_bits=8 * (mb - (total - block_params) * 2) / block_params,
               quant_s=n_quant_time, total_s=time.time() - t0)
    print(f"  {label:10s} | {mb/1e6:7.1f} MB | ppl {ppl:7.3f} | MMLU {acc:.3f} "
          f"[{lo:.2f},{hi:.2f}] | quantized in {n_quant_time:5.1f}s", flush=True)
    del wrapper, model
    return row


def real_7b_table():
    """The memory math the project is really about, for the model the guide asked for."""
    rows = []
    for name, params, layers, kv_heads, d_head in [
        ("Llama-2-7B", 6.74e9, 32, 32, 128),
        ("Llama-3-8B", 8.03e9, 32, 8, 128),
        ("Llama-3-70B", 70.6e9, 80, 8, 128),
    ]:
        for bits in (16, 8, 4):
            w = Q.packed_bytes(params, bits, 128) / 1e9 if bits < 16 else params * 2 / 1e9
            rows.append((name, bits, w))
    return rows


def plot(rows):
    labels = [r["label"] for r in rows]
    base = rows[0]

    # perplexity vs memory — the trade-off the whole field lives on
    fig, ax = ps.new_axes(7.4, 4.4)
    for i, r in enumerate(rows):
        ax.scatter(r["mb"], r["ppl"], s=90, color=ps.SERIES[i], zorder=3)
        dy = 10 if r["label"] != "INT4 RTN" else -18
        ax.annotate(f"{r['label']}\nppl {r['ppl']:.2f}", xy=(r["mb"], r["ppl"]),
                    xytext=(6, dy), textcoords="offset points",
                    color=ps.SERIES[i], fontsize=9)
    ax.axhline(base["ppl"], color=ps.BASELINE, ls="--", lw=1)
    ax.set_xlim(left=0)
    ps.finish(fig, ax, "Quality vs memory (Qwen2.5-0.5B, wikitext-2)",
              "model size (MB)", "perplexity (lower is better)",
              os.path.join(OUT, "quality_vs_memory.png"))

    # the 4-bit methods, head to head
    fig, ax = ps.new_axes(7.2, 4.2)
    four = [r for r in rows if r["bits"] == 4]
    x = range(len(four))
    ax.bar(x, [r["ppl"] for r in four],
           color=[ps.SERIES[2], ps.SERIES[1], ps.SERIES[0]], width=0.6, zorder=3)
    ax.axhline(base["ppl"], color=ps.INK_SECONDARY, ls="--", lw=1.2)
    ax.annotate(f"fp16 baseline ({base['ppl']:.2f})", xy=(len(four) - 0.5, base["ppl"]),
                xytext=(0, 6), textcoords="offset points", ha="right",
                color=ps.INK_SECONDARY, fontsize=9)
    for i, r in enumerate(four):
        ax.annotate(f"{r['ppl']:.2f}", xy=(i, r["ppl"]), xytext=(0, 4),
                    textcoords="offset points", ha="center",
                    color=ps.INK_SECONDARY, fontsize=10)
    ax.set_xticks(list(x)); ax.set_xticklabels([r["label"] for r in four])
    ax.set_ylim(0, max(r["ppl"] for r in four) * 1.25)
    ps.finish(fig, ax, "At 4 bits, how you round is worth more than the bits themselves",
              "", "perplexity (lower is better)", os.path.join(OUT, "four_bit_methods.png"))

    # MMLU with honest error bars
    fig, ax = ps.new_axes(7.2, 4.2)
    xs = range(len(rows))
    ax.bar(xs, [r["mmlu"] for r in rows],
           color=[ps.SERIES[i] for i in range(len(rows))], width=0.6, zorder=3)
    for i, r in enumerate(rows):
        ax.plot([i, i], [r["mmlu_lo"], r["mmlu_hi"]], color=ps.INK, lw=1.4, zorder=4)
    ax.axhline(0.25, color=ps.BASELINE, ls="--", lw=1)
    ax.annotate("chance", xy=(len(rows) - 0.6, 0.25), xytext=(0, 4),
                textcoords="offset points", color=ps.INK_MUTED, fontsize=9)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 0.65)
    ps.finish(fig, ax, "MMLU survives 4-bit — but the error bars are wide at n=171",
              "", "MMLU accuracy", os.path.join(OUT, "mmlu.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
    cache = os.path.join(CKPT, "results.json")

    if args.plot:
        plot(json.load(open(cache)))
        return

    text = Q.wikitext()
    mmlu_items = E.load_mmlu(per_subject=MMLU_PER_SUBJECT, seed=0)
    print(f"calibration: {N_CALIB}x{CALIB_LEN} tokens of wikitext | "
          f"perplexity: {N_PPL_SEQ}x{PPL_LEN} tokens | MMLU: {len(mmlu_items)} questions\n")

    rows = [evaluate(label, m, b, g, text, mmlu_items) for label, m, b, g in RECIPES]
    json.dump(rows, open(cache, "w"), indent=1)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "real_7b_memory.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "bits", "weights_GB"])
        w.writerows([(n, b, f"{gb:.2f}") for n, b, gb in real_7b_table()])

    print("\nwhat this would mean for the model the guide asked for:")
    for n, b, gb in real_7b_table():
        print(f"  {n:12s} {b:2d}-bit  {gb:6.2f} GB of weights")
    plot(rows)


if __name__ == "__main__":
    main()
