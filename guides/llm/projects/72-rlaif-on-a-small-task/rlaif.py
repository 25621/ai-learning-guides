"""RLAIF: preferences from an AI judge instead of a human, then DPO.

Human preference labels are the slow, expensive part of alignment. RLAIF asks:
can an AI judge label the pairs instead, and how close does the trained model
get for how few human labels?

We reuse the verifiable arithmetic task (project 28) so we have ground truth:

  * A pool of preference pairs, each a (correct, wrong) pair of completions.
    Half the wrong answers are random (easy to reject), half are near-misses,
    off by one or two (hard to reject).
  * GOLD ("human") preferences: the exact verifier labels every pair. This is
    the ceiling — but it costs one human label per pair.
  * AI preferences: a reward model, trained on only K human-labeled seed pairs,
    then labels the whole pool automatically. Human cost is just K.

We run DPO (project 33) from a shared SFT policy on each label set and read off
the resulting accuracy. Sweeping K traces quality against the human-label
budget: how cheap can the AI judge make alignment, and where does it plateau?

    python rlaif.py          # ~7 min on CPU
    python rlaif.py --plot   # redraw from outputs/rlaif.csv

Reuses sft_lib (28), the reward model (31) and the DPO loss (33).
"""

import copy
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "28-sft-a-1b-base-model"))
sys.path.insert(0, str(HERE.parent / "31-train-a-reward-model"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import sft_lib as L          # noqa: E402
import plot_style as ps      # noqa: E402
from rm import RewardModel   # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
BETA = 0.1


# --------------------------------------------------------------------------- #
# Preference pool: (correct, random-wrong) completions. Random-wrong rejects are
# the DPO-friendly case (project 33) — perfect labels lift accuracy cleanly, so
# the *only* thing that separates the AI judge from the gold verifier is how
# accurately the judge labels these pairs, which in turn depends on how many
# human seed labels it was trained on.
# --------------------------------------------------------------------------- #
def make_pool(n, rng):
    pool = []
    for _ in range(n):
        a, b, c = L.sample_problem(rng)
        w = rng.randint(0, 2 * L.MAXN)
        while w == c:
            w = rng.randint(0, 2 * L.MAXN)
        correct = L.seq_from(a, b, f"{c};")
        wrong = L.seq_from(a, b, f"{w};")
        pool.append({"correct": correct, "wrong": wrong})
    return pool


# --------------------------------------------------------------------------- #
# A reward-model AI judge trained on K human-labeled seed pairs. Fewer seeds =>
# a noisier judge, especially on near-misses.
# --------------------------------------------------------------------------- #
def train_judge(seed_pairs, steps, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    rm = RewardModel()
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    n = len(seed_pairs)
    g = random.Random(seed)
    for step in range(steps):
        batch = [seed_pairs[g.randrange(n)] for _ in range(min(64, n))]
        r_ch = rm.reward([p["correct"][0] for p in batch])
        r_rej = rm.reward([p["wrong"][0] for p in batch])
        loss = -F.logsigmoid(r_ch - r_rej).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
    return rm


@torch.no_grad()
def ai_label(rm, pool):
    """The AI judge picks which completion of each pair is 'chosen'.

    Returns (chosen, rejected) lists of (seq, mask), plus its labeling accuracy
    against ground truth (correct should be chosen)."""
    r_correct = rm.reward([p["correct"][0] for p in pool])
    r_wrong = rm.reward([p["wrong"][0] for p in pool])
    chosen, rejected, right = [], [], 0
    for p, rc, rw in zip(pool, r_correct, r_wrong):
        if rc >= rw:                          # judge prefers the correct one
            chosen.append(p["correct"]); rejected.append(p["wrong"]); right += 1
        else:                                 # judge got it backwards
            chosen.append(p["wrong"]); rejected.append(p["correct"])
    return chosen, rejected, right / len(pool)


def gold_label(pool):
    """The verifier labels every pair perfectly (correct is always chosen)."""
    return [p["correct"] for p in pool], [p["wrong"] for p in pool]


# --------------------------------------------------------------------------- #
# DPO (project 33's loss) from the shared SFT policy on a fixed label set.
# --------------------------------------------------------------------------- #
def dpo_loss(policy, ref, chosen, rejected, beta=BETA):
    cs, cm = [c[0] for c in chosen], [c[1] for c in chosen]
    rs, rmask = [r[0] for r in rejected], [r[1] for r in rejected]
    pi_w = L.completion_logprobs(policy, cs, cm)
    pi_l = L.completion_logprobs(policy, rs, rmask)
    with torch.no_grad():
        ref_w = L.completion_logprobs(ref, cs, cm)
        ref_l = L.completion_logprobs(ref, rs, rmask)
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(logits).mean()


def train_dpo(base, chosen, rejected, steps=300, lr=3e-4, seed=0):
    policy = copy.deepcopy(base)
    ref = copy.deepcopy(base)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95))
    g = random.Random(seed)
    n = len(chosen)
    for step in range(steps):
        idx = [g.randrange(n) for _ in range(64)]
        loss = dpo_loss(policy, ref, [chosen[i] for i in idx],
                        [rejected[i] for i in idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    return policy


def run():
    torch.set_num_threads(12)
    # A smaller operand range keeps the *judge* learnable from a realistic seed
    # budget (an RM must effectively learn to check the sum), so the AI-feedback
    # quality genuinely climbs with the number of human labels rather than
    # sitting at chance. The policy still starts only partially trained.
    L.MAXN = 15
    rng = random.Random(0)

    POOL_N = 2000
    pool = make_pool(POOL_N, rng)

    # Shared partially-trained SFT policy — the thing every DPO run improves on.
    # Kept short of the sharp grokking transition so DPO has room to improve.
    base = L.sft_model(steps=200)
    base_acc = L.accuracy(base, n=600)
    print(f"base SFT policy accuracy: {base_acc:.3f}", flush=True)

    # Ground-truth (gold) DPO: perfect labels, N human labels.
    ch, rj = gold_label(pool)
    gold_policy = train_dpo(base, ch, rj)
    gold_acc = L.accuracy(gold_policy, n=600)
    print(f"GOLD DPO ({POOL_N} human labels): acc {gold_acc:.3f}", flush=True)

    # AI-judge DPO, swept over the seed budget K.
    rows = [("condition", "human_labels", "judge_acc", "policy_acc")]
    rows.append(("base", 0, "", f"{base_acc:.4f}"))
    rows.append(("gold", POOL_N, "1.000", f"{gold_acc:.4f}"))

    ai_points = []
    for K in [16, 64, 128, 256, 512, 1024]:
        seed_pairs = make_pool(K, random.Random(100 + K))
        judge = train_judge(seed_pairs, steps=400)
        ch, rj, jacc = ai_label(judge, pool)
        policy = train_dpo(base, ch, rj)
        pacc = L.accuracy(policy, n=600)
        ai_points.append((K, jacc, pacc))
        rows.append((f"ai_K{K}", K, f"{jacc:.4f}", f"{pacc:.4f}"))
        print(f"AI DPO K={K:4d} human labels: judge-acc {jacc:.3f}  "
              f"policy-acc {pacc:.3f}", flush=True)

    with open(OUT / "rlaif.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    plot()


def plot():
    rows = list(csv.DictReader(open(OUT / "rlaif.csv")))
    base = next(float(r["policy_acc"]) for r in rows if r["condition"] == "base")
    gold = next(float(r["policy_acc"]) for r in rows if r["condition"] == "gold")
    gold_n = next(int(r["human_labels"]) for r in rows if r["condition"] == "gold")
    ai = [(int(r["human_labels"]), float(r["judge_acc"]), float(r["policy_acc"]))
          for r in rows if r["condition"].startswith("ai_")]
    ai.sort()

    # (a) quality vs human-label budget
    fig, ax = ps.new_axes(width=7.2, height=4.4)
    ks = [k for k, _, _ in ai]
    accs = [a for _, _, a in ai]
    ax.axhline(base, color=ps.INK_MUTED, ls="--", lw=1.3)
    ax.text(ks[0], base + 0.006, "SFT start (no preferences)",
            color=ps.INK_MUTED, fontsize=8)
    ax.axhline(gold, color=ps.SERIES[1], ls="--", lw=1.3)
    ax.text(ks[0], gold + 0.006, f"gold DPO ({gold_n} human labels)",
            color=ps.SERIES[1], fontsize=8)
    ax.plot(ks, accs, "-o", color=ps.SERIES[0], lw=2, ms=5,
            label="AI-judge DPO")
    ax.scatter([gold_n], [gold], color=ps.SERIES[1], s=70, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks + [gold_n])
    ax.set_xticklabels([str(k) for k in ks] + [str(gold_n)], fontsize=8)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ps.finish(fig, ax, "A cheap AI judge hurts; even a good one falls short of gold labels",
              "human labels spent (log scale)", "policy accuracy after DPO",
              OUT / "quality_vs_cost.png")

    # (b) judge accuracy vs policy accuracy
    fig, ax = ps.new_axes()
    jaccs = [j for _, j, _ in ai]
    ax.plot(ks, jaccs, "-o", color=ps.SERIES[3], lw=2, ms=5, label="judge label accuracy")
    ax.plot(ks, accs, "-o", color=ps.SERIES[0], lw=2, ms=5, label="resulting policy accuracy")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks], fontsize=8)
    ax.legend(frameon=False, loc="center right", fontsize=9)
    ps.finish(fig, ax, "The policy tracks the judge — but its label noise caps the gain",
              "reward-model seed labels K (log scale)", "accuracy",
              OUT / "judge_vs_policy.png")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    else:
        run()
