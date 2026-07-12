"""Load-test the mini engine: static batching vs continuous batching, and what paging buys.

The workload is the one that separates them: requests arrive at random times and
ask for wildly different numbers of tokens. Static batching makes every request in
a batch wait for the longest one *and* refuses to admit anyone new until the batch
drains. Continuous batching re-forms the batch every single decode step.

Run:  python3 serve.py           (~6 min)
      python3 serve.py --plot    (redraw from outputs/*.csv)
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

from engine import Engine, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "58-kv-cache-from-scratch"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))

from kv_lib import load_or_train  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoints")
torch.set_num_threads(12)

BLOCK = 16
MAX_MODEL_LEN = 512        # what a reserve-the-maximum engine must set aside per slot
N_REQUESTS = 48
RATE = 25.0                # arrivals per second (offered load)


def workload(data, n=N_REQUESTS, seed=0):
    """Poisson arrivals; short prompts; output lengths with a heavy tail.

    The tail is the point. Most requests want ~24 tokens, a few want ~160. In a
    static batch, the 24-token requests sit in the batch doing nothing until the
    160-token straggler finishes.
    """
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / RATE, n)
    arrivals = np.cumsum(gaps)
    reqs = []
    for i in range(n):
        p_len = int(rng.integers(24, 72))
        short = rng.random() < 0.8
        out_len = int(rng.integers(16, 32)) if short else int(rng.integers(120, 176))
        prompt = torch.randint(0, data.vocab_size, (p_len,))
        reqs.append(Sequence(sid=i, prompt=prompt, max_new=out_len,
                             arrival=float(arrivals[i])))
    return reqs


def percentiles(xs):
    a = np.array(xs) * 1000                                     # ms
    return dict(p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), mean=float(a.mean()))


def measure(model, data, policy, max_batch, seed=0):
    reqs = workload(data, seed=seed)
    eng = Engine(model, block_size=BLOCK, n_blocks=2048, max_batch=max_batch,
                 policy=policy)
    t0 = time.perf_counter()
    done = eng.run(reqs)
    wall = time.perf_counter() - t0

    out_tokens = sum(len(s.tokens) for s in done)
    ttft = [s.t_first - t0 - s.arrival + done[0].arrival for s in done]
    ttft = [max(t, 1e-6) for t in ttft]
    e2e = [s.t_end - t0 - s.arrival + done[0].arrival for s in done]

    # Memory: what paging actually held, vs what reserving the max would have needed.
    paged_blocks = eng.pool.peak_used
    reserve_blocks = eng.peak_running * (MAX_MODEL_LEN // BLOCK)
    row = dict(policy=policy, max_batch=max_batch, wall=wall,
               throughput=out_tokens / wall, out_tokens=out_tokens,
               ttft=percentiles(ttft), e2e=percentiles(e2e),
               peak_running=eng.peak_running, peak_tokens=eng.peak_tokens,
               paged_blocks=paged_blocks, reserve_blocks=reserve_blocks,
               waste=1 - eng.peak_tokens / max(1, paged_blocks * BLOCK))
    print(f"  {policy:11s} batch {max_batch:2d} | {wall:5.1f} s | "
          f"{row['throughput']:6.1f} tok/s | TTFT p50 {row['ttft']['p50']:7.0f} "
          f"p99 {row['ttft']['p99']:7.0f} ms | e2e p99 {row['e2e']['p99']:7.0f} ms | "
          f"KV {paged_blocks:4d} blocks (reserve-max would need {reserve_blocks})",
          flush=True)
    return row


def plot(rows, sweep):
    static = [r for r in rows if r["policy"] == "static"][0]
    cont = [r for r in rows if r["policy"] == "continuous"][0]

    # 1. throughput + latency percentiles, head to head
    fig, ax = ps.new_axes(7.4, 4.2)
    groups = ["TTFT p50", "TTFT p95", "TTFT p99", "e2e p99"]
    sv = [static["ttft"]["p50"], static["ttft"]["p95"], static["ttft"]["p99"],
          static["e2e"]["p99"]]
    cv = [cont["ttft"]["p50"], cont["ttft"]["p95"], cont["ttft"]["p99"],
          cont["e2e"]["p99"]]
    x = np.arange(len(groups))
    ax.bar(x - 0.19, sv, 0.36, color=ps.SERIES[2], label="static batching", zorder=3)
    ax.bar(x + 0.19, cv, 0.36, color=ps.SERIES[0], label="continuous batching", zorder=3)
    for i, (a, b) in enumerate(zip(sv, cv)):
        ax.annotate(f"{a/b:.1f}x", xy=(i, max(a, b)), xytext=(0, 5),
                    textcoords="offset points", ha="center",
                    color=ps.INK_SECONDARY, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=9)
    ps.finish(fig, ax, "Continuous batching pays off in the tail",
              "", "latency (ms)", os.path.join(OUT, "latency.png"))

    # 2. throughput vs max_batch
    fig, ax = ps.new_axes(7.2, 4.2)
    for i, pol in enumerate(["static", "continuous"]):
        pts = [(r["max_batch"], r["throughput"]) for r in sweep if r["policy"] == pol]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", lw=2,
                color=ps.SERIES[2 if pol == "static" else 0], label=pol)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylim(bottom=0)
    ps.finish(fig, ax, "Throughput comes from batching — but only if the batch stays full",
              "max concurrent sequences", "output tokens / s",
              os.path.join(OUT, "throughput.png"))

    # 3. KV memory: paged vs reserve-the-maximum
    fig, ax = ps.new_axes(7.2, 4.0)
    labels = ["reserve max_len\n(pre-PagedAttention)", "paged blocks\n(this engine)",
              "tokens actually live"]
    vals = [cont["reserve_blocks"] * BLOCK, cont["paged_blocks"] * BLOCK,
            cont["peak_tokens"]]
    ax.barh(labels[::-1], vals[::-1],
            color=[ps.SERIES[1], ps.SERIES[0], ps.SERIES[2]], zorder=3)
    for i, v in enumerate(vals[::-1]):
        ax.annotate(f"{v:,} token slots", xy=(v, i), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    color=ps.INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.35)
    ps.finish(fig, ax, "Paging cuts the KV reservation to what is really live",
              "KV cache (token slots at peak)", "",
              os.path.join(OUT, "kv_memory.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
    cache = os.path.join(CKPT, "results.json")

    if args.plot:
        d = json.load(open(cache))
        plot(d["head"], d["sweep"])
        return

    model, data = load_or_train()
    print(f"engine: {model.num_params()/1e6:.2f}M params | block size {BLOCK} tokens | "
          f"{N_REQUESTS} requests at {RATE:.0f} req/s\n")

    print("[1/2] same workload, same model, two schedulers (max_batch=16)")
    head = [measure(model, data, p, 16) for p in ("static", "continuous")]

    print("\n[2/2] throughput vs how many sequences we let run at once")
    sweep = [measure(model, data, p, b)
             for b in (2, 4, 8, 16, 32) for p in ("static", "continuous")]

    json.dump(dict(head=head, sweep=sweep), open(cache, "w"), indent=1)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "max_batch", "wall_s", "throughput_tok_s", "ttft_p50_ms",
                    "ttft_p95_ms", "ttft_p99_ms", "e2e_p50_ms", "e2e_p99_ms",
                    "peak_running", "peak_tokens", "paged_blocks", "reserve_blocks"])
        for r in head + sweep:
            w.writerow([r["policy"], r["max_batch"], f"{r['wall']:.2f}",
                        f"{r['throughput']:.1f}", f"{r['ttft']['p50']:.0f}",
                        f"{r['ttft']['p95']:.0f}", f"{r['ttft']['p99']:.0f}",
                        f"{r['e2e']['p50']:.0f}", f"{r['e2e']['p99']:.0f}",
                        r["peak_running"], r["peak_tokens"], r["paged_blocks"],
                        r["reserve_blocks"]])
    plot(head, sweep)


if __name__ == "__main__":
    main()
