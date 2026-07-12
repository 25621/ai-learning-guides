"""Speculative decoding: let the small model guess, make the big model check.

The guide's pairing is a 1B draft with a 7B target. Neither fits this CPU box, so
we run the same algorithm on two *real* pairs that share a tokenizer — which is the
one hard requirement, since the draft's token ids must mean the same thing to the
target:

  SmolLM2  360M target / 135M draft   — same family, same data: guesses land often
  GPT-2    774M target / 124M draft   — a genuinely cheap draft: guesses cost little

Those two pairs fail in opposite directions, and that is the lesson. Speculative
decoding has exactly two levers:

    acceptance rate  a   — how often the draft guesses what the target would say
    cost ratio       c   — draft forward time / target forward time

    expected tokens per round   E = (1 - a^(k+1)) / (1 - a)
    cost per round (target fwd) C = k*c + 1
    speedup                       = E / C

A draft that agrees with the target but is not much cheaper (SmolLM) buys nothing.
A cheap draft that disagrees (GPT-2) buys little either. You need both — which is
why production stacks use a ~10x smaller draft *distilled from the target*.

Run:  python3 speculative.py           (~7 min)
      python3 speculative.py --plot    (redraw from outputs/results.json)
"""

import argparse
import csv
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-train-a-bpe-from-scratch"))
import plot_style as ps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoints")
torch.set_num_threads(12)

PAIRS = {
    "SmolLM2 360M / 135M": ("HuggingFaceTB/SmolLM2-360M-Instruct",
                            "HuggingFaceTB/SmolLM2-135M-Instruct"),
    "GPT-2 774M / 124M": ("gpt2-large", "gpt2"),
}
K_VALUES = [1, 2, 4, 6]
N_NEW = 48
PROMPTS = [
    "The capital of France is Paris, and the city is famous for",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
    "The best way to learn programming is to",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
]


# ------------------------------------------------------------------- the baseline
@torch.no_grad()
def greedy(model, ids, n_new):
    """Ordinary cached greedy decoding — one target forward per token."""
    out = model(ids, use_cache=True)
    past, toks = out.past_key_values, []
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    toks.append(int(nxt))
    for _ in range(n_new - 1):
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(int(nxt))
    return toks


# --------------------------------------------------------------- the real thing
@torch.no_grad()
def speculative(target, draft, ids, n_new, k):
    """Draft k tokens, verify them all in one target pass, keep the longest match.

    Bookkeeping that makes this correct: both caches hold every *committed* token
    except the newest one (`cur`), which is fed in at the front of the next pass.
    When the target rejects a guess, both caches are cropped back — a rejected
    token must leave no trace, or the next round attends to a token that was never
    really generated.
    """
    t_out = target(ids, use_cache=True)
    t_past = t_out.past_key_values
    d_out = draft(ids, use_cache=True)
    d_past = d_out.past_key_values

    cur = t_out.logits[:, -1].argmax(-1, keepdim=True)   # first token: free, from prefill
    committed = [int(cur)]
    n_cached = ids.shape[1]                              # tokens currently in both caches
    proposed = accepted = rounds = 0

    while len(committed) < n_new:
        # ---- 1. the draft runs ahead, k tokens, autoregressively --------------
        drafts, d_in = [], cur
        for _ in range(k):
            d_out = draft(d_in, past_key_values=d_past, use_cache=True)
            d_past = d_out.past_key_values
            d_in = d_out.logits[:, -1].argmax(-1, keepdim=True)
            drafts.append(int(d_in))

        # ---- 2. the target checks all k in ONE forward pass --------------------
        block = torch.cat([cur, torch.tensor([drafts])], dim=1)     # (1, k+1)
        t_out = target(block, past_key_values=t_past, use_cache=True)
        t_past = t_out.past_key_values
        t_pred = t_out.logits[0].argmax(-1).tolist()     # what the target would have said

        # ---- 3. accept the longest prefix the target agrees with ---------------
        j = 0
        while j < k and drafts[j] == t_pred[j]:
            j += 1
        bonus = t_pred[j]                                # the target's own next token

        committed += drafts[:j] + [bonus]
        proposed += k
        accepted += j
        rounds += 1

        # ---- 4. roll the caches back to exactly the committed tokens -----------
        n_cached += j + 1                                # cur + the j accepted drafts
        t_past.crop(n_cached)
        d_past.crop(n_cached)
        cur = torch.tensor([[bonus]])

    return committed[:n_new], dict(proposed=proposed, accepted=accepted, rounds=rounds,
                                   alpha=accepted / max(1, proposed),
                                   tokens_per_round=len(committed) / max(1, rounds))


# --------------------------------------------------------------------- the model
def predicted_speedup(alpha, k, c):
    """E[tokens per round] / cost per round, both in units of one target forward."""
    e = (1 - alpha ** (k + 1)) / (1 - alpha) if alpha < 1 else k + 1
    return e / (k * c + 1)


def timed_greedy(model, tok, n_new):
    """Greedy-decode every prompt and return (outputs, seconds)."""
    outs, t = {}, 0.0
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        t0 = time.perf_counter()
        outs[p] = greedy(model, ids, n_new)
        t += time.perf_counter() - t0
    return outs, t


def run_pair(name, target_id, draft_id):
    print(f"\n=== {name} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(target_id)
    target = AutoModelForCausalLM.from_pretrained(target_id, dtype=torch.float32).eval()
    draft = AutoModelForCausalLM.from_pretrained(draft_id, dtype=torch.float32).eval()

    # Warm the kernels, then time each model doing the *same* job on its own. The
    # ratio of those two timings is exactly the `c` the speedup formula wants —
    # no synthetic single-step microbenchmark needed.
    timed_greedy(draft, tok, 4)
    timed_greedy(target, tok, 4)

    baselines, base_time = timed_greedy(target, tok, N_NEW)
    _, draft_time = timed_greedy(draft, tok, N_NEW)
    c = draft_time / base_time

    rows, exact = [], True
    print(f"  baseline (plain cached greedy): {base_time:5.1f} s "
          f"({base_time/(len(PROMPTS)*N_NEW)*1000:.0f} ms/token)", flush=True)
    print(f"  the draft alone would take {draft_time:5.1f} s  =>  cost ratio c = {c:.3f}",
          flush=True)

    for k in K_VALUES:
        t0 = time.perf_counter()
        acc = prop = 0
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids
            toks, st = speculative(target, draft, ids, N_NEW, k)
            acc += st["accepted"]; prop += st["proposed"]
            if toks != baselines[p]:
                exact = False                            # must be bit-identical to greedy
        dt = time.perf_counter() - t0
        alpha = acc / prop
        row = dict(pair=name, k=k, alpha=alpha, wall=dt, speedup=base_time / dt,
                   c=c, predicted=predicted_speedup(alpha, k, c))
        rows.append(row)
        print(f"  k={k} | acceptance {alpha:.3f} | {dt:5.1f} s | speedup {row['speedup']:.2f}x "
              f"| model predicts {row['predicted']:.2f}x", flush=True)

    print(f"  output identical to plain greedy: {exact}")
    del target, draft
    return rows, base_time, c, exact


# ------------------------------------------------------------------------ report
def plot(rows):
    pairs = list(dict.fromkeys(r["pair"] for r in rows))

    # measured vs predicted speedup
    fig, ax = ps.new_axes(7.4, 4.3)
    for i, pair in enumerate(pairs):
        rs = [r for r in rows if r["pair"] == pair]
        ax.plot([r["k"] for r in rs], [r["speedup"] for r in rs], "o-", lw=2,
                color=ps.SERIES[i], label=f"{pair}  (c={rs[0]['c']:.2f})")
        ax.plot([r["k"] for r in rs], [r["predicted"] for r in rs], "--", lw=1.4,
                color=ps.SERIES[i], alpha=0.55)
    # what the guide's 7B/1B pair would do, using the same formula
    ks = K_VALUES
    ax.plot(ks, [predicted_speedup(0.75, k, 0.14) for k in ks], ":", lw=2,
            color=ps.SERIES[2], label="7B target / 1B draft (a=0.75, c=0.14), predicted")
    ax.axhline(1.0, color=ps.BASELINE, ls="-", lw=1)
    ax.annotate("no speedup", xy=(ks[-1], 1.0), xytext=(-4, 5),
                textcoords="offset points", ha="right", color=ps.INK_MUTED, fontsize=9)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=8.5)
    ax.set_ylim(bottom=0)
    ps.finish(fig, ax, "Speedup needs a draft that is both cheap and right (dashed = formula)",
              "k (tokens drafted per round)", "wall-clock speedup vs greedy",
              os.path.join(OUT, "speedup.png"))

    # acceptance rate
    fig, ax = ps.new_axes(7.2, 4.0)
    for i, pair in enumerate(pairs):
        rs = [r for r in rows if r["pair"] == pair]
        ax.plot([r["k"] for r in rs], [r["alpha"] for r in rs], "o-", lw=2,
                color=ps.SERIES[i], label=pair)
        ax.annotate(f"{rs[0]['alpha']:.0%}", xy=(rs[0]["k"], rs[0]["alpha"]),
                    xytext=(6, 4), textcoords="offset points",
                    color=ps.SERIES[i], fontsize=9)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylim(0, 1)
    ps.finish(fig, ax, "Acceptance rate: how often the draft says what the target would",
              "k (tokens drafted per round)", "fraction of drafted tokens accepted",
              os.path.join(OUT, "acceptance.png"))

    # the design space: speedup as a function of both levers
    fig, ax = ps.new_axes(7.2, 4.3)
    for i, cc in enumerate([0.05, 0.14, 0.3, 0.5]):
        alphas = [a / 100 for a in range(30, 100)]
        ax.plot(alphas, [max(predicted_speedup(a, 4, cc) for a in [a]) for a in alphas],
                lw=2, color=ps.SERIES[i], label=f"c = {cc}")
    for i, pair in enumerate(pairs):
        rs = [r for r in rows if r["pair"] == pair and r["k"] == 4]
        if rs:
            ax.scatter([rs[0]["alpha"]], [rs[0]["speedup"]], s=80, zorder=5,
                       color=ps.INK, marker="x")
            ax.annotate(pair, xy=(rs[0]["alpha"], rs[0]["speedup"]), xytext=(-8, 10),
                        textcoords="offset points", ha="right",
                        color=ps.INK_SECONDARY, fontsize=8.5)
    ax.axhline(1.0, color=ps.BASELINE, lw=1)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=9, title="draft cost")
    ps.finish(fig, ax, "The design space at k=4 (x = our two measured pairs)",
              "acceptance rate", "predicted speedup",
              os.path.join(OUT, "design_space.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
    cache = os.path.join(CKPT, "results.json")

    if args.plot:
        plot(json.load(open(cache))["rows"])
        return

    all_rows, meta = [], {}
    for name, (t_id, d_id) in PAIRS.items():
        rows, base, c, exact = run_pair(name, t_id, d_id)
        all_rows += rows
        meta[name] = dict(baseline_s=base, c=c, exact=exact)

    json.dump(dict(rows=all_rows, meta=meta), open(cache, "w"), indent=1)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "k", "acceptance", "cost_ratio_c", "wall_s",
                    "speedup_measured", "speedup_predicted"])
        for r in all_rows:
            w.writerow([r["pair"], r["k"], f"{r['alpha']:.3f}", f"{r['c']:.3f}",
                        f"{r['wall']:.1f}", f"{r['speedup']:.3f}", f"{r['predicted']:.3f}"])
    plot(all_rows)


if __name__ == "__main__":
    main()
