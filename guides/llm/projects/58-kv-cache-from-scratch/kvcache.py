"""KV cache from scratch: prove it is exact, then measure what it buys.

Three questions, in the order an engineer should ask them:

  1. correctness — does cached decoding produce *exactly* the naive tokens?
  2. speed      — how does per-token latency grow with context, with and without?
  3. memory     — what does the cache cost, and what would it cost on a real model?

Run:  python3 kvcache.py          (~6 min: ~3 min to train the shared model, then benchmarks)
      python3 kvcache.py --plot   (re-draw figures from outputs/*.csv)
"""

import argparse
import csv
import os
import sys
import time

import torch

import kv_lib
from kv_lib import KVCache, forward_cached, generate_cached, generate_naive, load_or_train

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-train-a-bpe-from-scratch"))
import plot_style as ps  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
torch.set_num_threads(12)


# --------------------------------------------------------------------- 1. correct
def check_correctness(model, data):
    """Cached and naive decoding must agree token-for-token, not just 'look similar'."""
    prompt = data.encode("KING RICHARD III:\nNow is the winter of our discontent")
    n = 64

    greedy_naive = generate_naive(model, prompt, n, temperature=0.0)
    greedy_cached = generate_cached(model, prompt, n, temperature=0.0)
    same = torch.equal(greedy_naive, greedy_cached)

    # And the logits themselves, not merely the argmax, should match to float noise.
    logits_naive, _ = model(greedy_naive[:, :-1])
    cache = KVCache(model.cfg, 1, greedy_naive.size(1), prompt.device)
    logits_cached = forward_cached(model, greedy_naive[:, :-1], cache)
    max_diff = (logits_naive - logits_cached).abs().max().item()

    print(f"  greedy tokens identical : {same}")
    print(f"  max |logit| difference  : {max_diff:.2e}")
    text = data.decode(greedy_cached[0, prompt.size(1):].tolist())
    print(f"  sample continuation     : {text[:60]!r}")
    return same, max_diff, text


# ----------------------------------------------------------------------- 2. speed
def fwd_flops(cfg, n_params, T):
    """FLOPs for one forward pass over T tokens: matmuls (2ND) + attention (T^2)."""
    d_head = cfg.n_embd // cfg.n_head
    return 2 * n_params * T + 4 * cfg.n_layer * cfg.n_head * d_head * T * T


def token_flops(cfg, n_params, T):
    """FLOPs to produce *one more* token at context T, naive vs cached."""
    d_head = cfg.n_embd // cfg.n_head
    naive = fwd_flops(cfg, n_params, T)                       # redo the whole prefix
    cached = 2 * n_params + 4 * cfg.n_layer * cfg.n_head * d_head * T   # one query, T keys
    return naive, cached


def bench_decode(model, data, prefix_lens, n_new=32, repeats=5):
    """Time one decoded token at a given context length, cached vs naive.

    The naive path pays for the whole prefix on every token; the cached path pays
    for one token plus an attention read over the cache. Timings are the *minimum*
    of several repeats — on a shared CPU the minimum is the honest estimate; the
    mean mostly measures whatever else the machine was doing.
    """
    rows = []
    for L in prefix_lens:
        idx = torch.randint(0, data.vocab_size, (1, L))
        generate_naive(model, idx, 2, temperature=0.0)        # warm up the kernels
        generate_cached(model, idx, 2, temperature=0.0)

        best_naive = best_cached = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            generate_naive(model, idx, n_new, temperature=0.0)
            best_naive = min(best_naive, (time.perf_counter() - t0) / n_new)

            # Exclude prefill from the cached number so we are comparing like with
            # like: the cost of *one more decoded token* at context length L.
            cache = KVCache(model.cfg, 1, L + n_new)
            logits = forward_cached(model, idx, cache)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            t0 = time.perf_counter()
            for _ in range(n_new):
                logits = forward_cached(model, nxt, cache)
                nxt = logits[:, -1].argmax(-1, keepdim=True)
            best_cached = min(best_cached, (time.perf_counter() - t0) / n_new)

        fn, fc = token_flops(model.cfg, model.num_params(), L)
        rows.append((L, best_naive * 1000, best_cached * 1000, best_naive / best_cached, fn / fc))
        print(f"  ctx {L:4d} | naive {best_naive*1000:7.2f} ms/tok | "
              f"cached {best_cached*1000:6.2f} ms/tok | {best_naive/best_cached:5.2f}x wall | "
              f"{fn/fc:6.1f}x FLOPs", flush=True)
    return rows


def bench_generation(model, data, lengths, prompt_len=32, repeats=3):
    """End-to-end: total wall-clock to generate N tokens from a short prompt."""
    rows = []
    for n in lengths:
        idx = torch.randint(0, data.vocab_size, (1, prompt_len))
        generate_naive(model, idx, 4, temperature=0.0)        # warm up
        generate_cached(model, idx, 4, temperature=0.0)
        tn = tc = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            generate_naive(model, idx, n, temperature=0.0)
            tn = min(tn, time.perf_counter() - t0)
            t0 = time.perf_counter()
            generate_cached(model, idx, n, temperature=0.0)
            tc = min(tc, time.perf_counter() - t0)
        rows.append((n, tn, tc, tn / tc))
        print(f"  generate {n:4d} tokens | naive {tn:6.2f} s | cached {tc:5.2f} s | "
              f"{tn/tc:5.2f}x", flush=True)
    return rows


# ---------------------------------------------------------------------- 3. memory
def kv_bytes(n_layer, n_kv_heads, d_head, seq_len, batch=1, bytes_per=2):
    """2 (K and V) x layers x kv-heads x d_head x tokens x batch x bytes-per-number."""
    return 2 * n_layer * n_kv_heads * d_head * seq_len * batch * bytes_per


def memory_table(cfg):
    """Our toy next to the models the formula actually matters for."""
    rows = [("this project (1.8M)", cfg.n_layer, cfg.n_kv_heads,
             cfg.n_embd // cfg.n_head, 512, 1),
            ("Llama-3-8B  @ 8k",  32, 8,  128, 8192, 1),
            ("Llama-3-8B  @ 8k, batch 32", 32, 8, 128, 8192, 32),
            ("Llama-3-70B @ 8k, batch 32", 80, 8, 128, 8192, 32),
            ("Llama-3-70B @ 128k, batch 32", 80, 8, 128, 131072, 32)]
    out = []
    for name, L, kv, dh, T, B in rows:
        gb = kv_bytes(L, kv, dh, T, B) / 1e9
        out.append((name, L, kv, dh, T, B, gb))
        print(f"  {name:30s} {gb*1000:9.1f} MB")
    return out


# ------------------------------------------------------------------------- report
def plot(decode_rows, gen_rows):
    # per-token latency vs context length
    fig, ax = ps.new_axes()
    L = [r[0] for r in decode_rows]
    ax.plot(L, [r[1] for r in decode_rows], "o-", color=ps.SERIES[2], lw=2, label="naive")
    ax.plot(L, [r[2] for r in decode_rows], "o-", color=ps.SERIES[0], lw=2, label="KV cache")
    ax.annotate("naive: cost grows with the prefix\n(the whole context, every token)",
                xy=(L[-1], decode_rows[-1][1]), xytext=(-16, -34), textcoords="offset points",
                ha="right", color=ps.SERIES[2], fontsize=9)
    ax.annotate("cached: flat — one token's work", xy=(L[-1], decode_rows[-1][2]),
                xytext=(-10, 14), textcoords="offset points", ha="right",
                color=ps.SERIES[0], fontsize=9)
    ax.set_ylim(bottom=0)
    ps.finish(fig, ax, "Cost of one decoded token", "context length (tokens)",
              "latency (ms / token)", os.path.join(OUT, "decode_latency.png"))

    # total generation time + speedup
    fig, ax = ps.new_axes()
    n = [r[0] for r in gen_rows]
    ax.plot(n, [r[1] for r in gen_rows], "o-", color=ps.SERIES[2], lw=2, label="naive")
    ax.plot(n, [r[2] for r in gen_rows], "o-", color=ps.SERIES[0], lw=2, label="KV cache")
    for x, tn, tc, sp in gen_rows:
        ax.annotate(f"{sp:.1f}x", xy=(x, tn), xytext=(0, 6), textcoords="offset points",
                    ha="center", color=ps.INK_SECONDARY, fontsize=8)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylim(bottom=0)
    ps.finish(fig, ax, "Time to generate a continuation (quadratic vs linear)",
              "tokens generated", "wall-clock (s)", os.path.join(OUT, "generation_time.png"))

    # KV cache memory growth for real models
    fig, ax = ps.new_axes()
    T = [2 ** i for i in range(9, 18)]
    for i, (name, L_, kv, dh) in enumerate([("Llama-3-8B (GQA, 8 kv heads)", 32, 8, 128),
                                            ("Llama-3-70B (GQA, 8 kv heads)", 80, 8, 128),
                                            ("70B with MHA (64 kv heads)", 80, 64, 128)]):
        ax.plot(T, [kv_bytes(L_, kv, dh, t, 32) / 1e9 for t in T], "o-",
                color=ps.SERIES[i], lw=2, label=name, ms=3)
    ax.axhline(80, color=ps.SERIES[2], ls="--", lw=1)
    ax.annotate("one H100 (80 GB)", xy=(T[0], 80), xytext=(0, 5),
                textcoords="offset points", color=ps.SERIES[2], fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=8, loc="upper left")
    ps.finish(fig, ax, "What the cache costs at scale (batch 32, fp16)",
              "context length (tokens)", "KV cache (GB)",
              os.path.join(OUT, "kv_memory.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="redraw figures from outputs/*.csv")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.plot:
        d = list(csv.reader(open(os.path.join(OUT, "decode_latency.csv"))))[1:]
        g = list(csv.reader(open(os.path.join(OUT, "generation_time.csv"))))[1:]
        plot([(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in d],
             [(int(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in g])
        return

    model, data = load_or_train()
    print(f"\nmodel: {model.num_params()/1e6:.2f}M params, {model.cfg.n_layer} layers, "
          f"context {model.cfg.block_size}\n")

    print("[1/3] correctness — cached decoding must be exact, not merely close")
    same, max_diff, sample = check_correctness(model, data)

    print("\n[2/3] speed — latency of one decoded token vs context length")
    decode_rows = bench_decode(model, data, [64, 128, 192, 256, 320, 384, 448, 512])

    print("\n      end-to-end generation from a 32-token prompt")
    gen_rows = bench_generation(model, data, [32, 64, 128, 256, 384])

    print("\n[3/3] memory — the cache is not free")
    mem = memory_table(model.cfg)

    with open(os.path.join(OUT, "decode_latency.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ctx_len", "naive_ms", "cached_ms", "speedup_wall", "speedup_flops"])
        w.writerows([(a, f"{b:.4f}", f"{c:.4f}", f"{d:.4f}", f"{e:.2f}")
                     for a, b, c, d, e in decode_rows])
    with open(os.path.join(OUT, "generation_time.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["n_tokens", "naive_s", "cached_s", "speedup"])
        w.writerows([(a, f"{b:.4f}", f"{c:.4f}", f"{d:.4f}") for a, b, c, d in gen_rows])
    with open(os.path.join(OUT, "kv_memory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "layers", "kv_heads", "d_head", "seq_len", "batch", "kv_GB"])
        w.writerows([(n, L, kv, dh, T, B, f"{gb:.4f}") for n, L, kv, dh, T, B, gb in mem])
    with open(os.path.join(OUT, "correctness.txt"), "w") as f:
        f.write(f"greedy tokens identical: {same}\n")
        f.write(f"max abs logit difference: {max_diff:.3e}\n")
        f.write(f"sample continuation:\n{sample}\n")

    plot(decode_rows, gen_rows)


if __name__ == "__main__":
    main()
