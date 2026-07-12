"""Prefix caching: the same workload, the same engine, one flag.

Real traffic almost never sends fresh text. It sends the *same* long system
prompt — tool definitions, retrieved documents, a persona, few-shot examples —
followed by a short user turn. Without a prefix cache every request re-runs that
boilerplate through every layer. With one, the first request pays and everyone
after it reads the keys and values straight out of the block pool.

This project runs project 61's engine twice over an identical workload, flipping
`prefix_cache`, and reports what the cache does to TTFT (the metric a user
actually feels) as the shared prefix grows.

Run:  python3 prefix.py           (~5 min)
      python3 prefix.py --plot    (redraw from outputs/*.csv)
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "61-serve-with-vllm"))
sys.path.insert(0, os.path.join(HERE, "..", "58-kv-cache-from-scratch"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))

from engine import Engine, Sequence  # noqa: E402
from kv_lib import load_or_train  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoints")
torch.set_num_threads(12)

BLOCK = 16
N_REQUESTS = 32
RATE = 20.0
OUT_TOKENS = 24                 # short replies — the realistic chat shape
PREFIX_LENS = [0, 64, 128, 256]
N_TENANTS = 3                   # three different system prompts share the server


def workload(data, prefix_len, n=N_REQUESTS, seed=0):
    """`n` requests, each = [one of N_TENANTS system prompts] + [a unique user turn].

    The system prompts are real text (Shakespeare, the corpus the model knows) so
    the prefill is representative work, not noise.
    """
    rng = np.random.default_rng(seed)
    text = data.train
    systems = [text[i * 1000: i * 1000 + prefix_len].clone() for i in range(N_TENANTS)]

    reqs = []
    arrivals = np.cumsum(rng.exponential(1.0 / RATE, n))
    for i in range(n):
        sys_ids = systems[i % N_TENANTS] if prefix_len else torch.empty(0, dtype=torch.long)
        user = torch.randint(0, data.vocab_size, (int(rng.integers(8, 24)),))
        prompt = torch.cat([sys_ids, user])
        reqs.append(Sequence(sid=i, prompt=prompt, max_new=OUT_TOKENS,
                             arrival=float(arrivals[i])))
    return reqs


def measure(model, data, prefix_len, cache_on, seed=0):
    reqs = workload(data, prefix_len, seed=seed)
    eng = Engine(model, block_size=BLOCK, n_blocks=4096, max_batch=16,
                 policy="continuous", prefix_cache=cache_on)
    t0 = time.perf_counter()
    done = eng.run(reqs)
    wall = time.perf_counter() - t0

    origin = min(s.arrival for s in done)
    ttft = np.array([max(s.t_first - t0 - (s.arrival - origin), 1e-6) for s in done]) * 1000
    e2e = np.array([s.t_end - t0 - (s.arrival - origin) for s in done]) * 1000
    out_tokens = sum(len(s.tokens) for s in done)

    row = dict(prefix_len=prefix_len, cache=cache_on, wall=wall,
               throughput=out_tokens / wall,
               ttft_p50=float(np.percentile(ttft, 50)),
               ttft_p95=float(np.percentile(ttft, 95)),
               ttft_p99=float(np.percentile(ttft, 99)),
               e2e_p99=float(np.percentile(e2e, 99)),
               prefill_tokens=eng.prefill_tokens, saved_tokens=eng.saved_tokens,
               reused=sum(s.reused_blocks for s in done))
    print(f"  prefix {prefix_len:3d} | cache {'on ' if cache_on else 'off'} | "
          f"TTFT p50 {row['ttft_p50']:6.0f} p99 {row['ttft_p99']:6.0f} ms | "
          f"{row['throughput']:5.1f} tok/s | prefilled {eng.prefill_tokens:5d} tokens "
          f"(skipped {eng.saved_tokens:5d})", flush=True)
    return row


def plot(rows):
    on = [r for r in rows if r["cache"]]
    off = [r for r in rows if not r["cache"]]
    xs = [r["prefix_len"] for r in off]

    # 1. TTFT vs shared-prefix length
    fig, ax = ps.new_axes(7.4, 4.3)
    ax.plot(xs, [r["ttft_p50"] for r in off], "o-", lw=2, color=ps.SERIES[2],
            label="cache off — p50")
    ax.plot(xs, [r["ttft_p99"] for r in off], "o--", lw=1.6, color=ps.SERIES[2],
            alpha=0.6, label="cache off — p99")
    ax.plot(xs, [r["ttft_p50"] for r in on], "o-", lw=2, color=ps.SERIES[0],
            label="cache on — p50")
    ax.plot(xs, [r["ttft_p99"] for r in on], "o--", lw=1.6, color=ps.SERIES[0],
            alpha=0.6, label="cache on — p99")
    last_off, last_on = off[-1], on[-1]
    ax.annotate(f"{last_off['ttft_p50']/max(last_on['ttft_p50'],1e-6):.1f}x faster\nto first token",
                xy=(xs[-1], last_on["ttft_p50"]), xytext=(-10, 26),
                textcoords="offset points", ha="right", color=ps.SERIES[0], fontsize=9)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=8.5)
    ax.set_ylim(bottom=0)
    ps.finish(fig, ax, "The longer the shared system prompt, the more the cache saves",
              "shared system-prompt length (tokens)", "time to first token (ms)",
              os.path.join(OUT, "ttft.png"))

    # 2. prefill work actually done
    fig, ax = ps.new_axes(7.2, 4.2)
    w = 0.36
    x = np.arange(len(xs))
    ax.bar(x - w / 2, [r["prefill_tokens"] for r in off], w, color=ps.SERIES[2],
           label="cache off", zorder=3)
    ax.bar(x + w / 2, [r["prefill_tokens"] for r in on], w, color=ps.SERIES[0],
           label="cache on", zorder=3)
    for i, r in enumerate(on):
        if r["saved_tokens"]:
            ax.annotate(f"-{r['saved_tokens'] / (r['saved_tokens'] + r['prefill_tokens']):.0%}",
                        xy=(i + w / 2, r["prefill_tokens"]), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        color=ps.SERIES[0], fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(xs)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=9)
    ps.finish(fig, ax, "Prefill tokens the server actually had to compute",
              "shared system-prompt length (tokens)", "prompt tokens prefilled",
              os.path.join(OUT, "prefill_work.png"))

    # 3. throughput
    fig, ax = ps.new_axes(7.2, 4.0)
    ax.plot(xs, [r["throughput"] for r in off], "o-", lw=2, color=ps.SERIES[2],
            label="cache off")
    ax.plot(xs, [r["throughput"] for r in on], "o-", lw=2, color=ps.SERIES[0],
            label="cache on")
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylim(bottom=0)
    ps.finish(fig, ax, "Throughput: the prefill the cache skips is time back on the clock",
              "shared system-prompt length (tokens)", "output tokens / s",
              os.path.join(OUT, "throughput.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
    cache_file = os.path.join(CKPT, "results.json")

    if args.plot:
        plot(json.load(open(cache_file)))
        return

    model, data = load_or_train()
    print(f"{N_REQUESTS} requests at {RATE:.0f}/s | {N_TENANTS} shared system prompts | "
          f"{OUT_TOKENS} output tokens each | block size {BLOCK}\n")

    rows = []
    for p in PREFIX_LENS:
        for cache_on in (False, True):
            rows.append(measure(model, data, p, cache_on))

    json.dump(rows, open(cache_file, "w"), indent=1)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    plot(rows)


if __name__ == "__main__":
    main()
