"""Arena reproduction: rank five models by who beats whom, then shake the table.

Five open models answer the same 30 prompts. An LLM judge plays every pair on
every prompt (both presentation orders, to cancel position bias), and the
win/loss record becomes an Elo rating — the Chatbot-Arena recipe in miniature.

The point of *reproducing* it is to feel how much the ranking wobbles:

  * SEED STABILITY  — Elo depends on the order matches are processed. We bootstrap
    over 200 random orderings and report each model's rating distribution.
  * JUDGE STABILITY — we recompute the whole ranking with a *second* judge model
    and check whether the order survives. A ranking that flips when you change
    the judge is a ranking of the judge, not the models.

Generation is cached to `checkpoints/`, so re-running the analysis is instant.

CPU, ~7 min the first time (mostly the five models generating).
"""

import csv
import itertools
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "51-mmlu-re-run"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))
import plot_style as ps  # noqa: E402
from eval_lib import Model  # noqa: E402

# Chosen to span a real quality range so the ranking is meaningful, and all
# small enough to generate on a CPU.
CONTESTANTS = [
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ("bloom-560m", "bigscience/bloom-560m"),
    ("gpt2", "gpt2"),
]
JUDGE_A = "Qwen/Qwen2.5-0.5B-Instruct"
JUDGE_B = "HuggingFaceTB/SmolLM2-360M-Instruct"
N_PROMPTS = 20      # keeps 10 pairs x 20 prompts x 2 orders x 2 judges under budget
MAX_NEW = 48
OUT = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoints")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

ALL_PROMPTS = [
    "Explain what a black hole is in two sentences.",
    "Write a short, cheerful haiku about coffee.",
    "What are three tips for saving money on groceries?",
    "Explain the difference between weather and climate.",
    "Give me a simple recipe for scrambled eggs.",
    "Why is the sky blue? Keep it brief.",
    "Suggest a name for a friendly robot vacuum and say why.",
    "What should I pack for a weekend camping trip?",
    "Summarize the plot of Cinderella in three sentences.",
    "How do I stay motivated while learning to code?",
    "Explain photosynthesis to a ten-year-old.",
    "What are the benefits of regular exercise?",
    "Write a one-line pep talk for someone before a job interview.",
    "Describe the taste of a mango to someone who has never had one.",
    "What is the capital of Japan and one fact about it?",
    "Give two reasons recycling matters.",
    "How does a rainbow form?",
    "Recommend a book for a rainy afternoon and say why.",
    "What's a polite way to decline a party invitation?",
    "Explain what 'compound interest' means with a small example.",
    "List three uses for a paperclip besides holding paper.",
    "What causes the seasons to change?",
    "Write a two-sentence bedtime story about a sleepy dragon.",
    "How can I make my small apartment feel bigger?",
    "What is machine learning, in one plain sentence?",
    "Suggest a healthy snack and why it's good for you.",
    "Explain why ice floats on water.",
    "Give a beginner's tip for taking better phone photos.",
    "What's the difference between a fruit and a vegetable?",
    "Write an encouraging note for someone starting a new job.",
]
PROMPTS = ALL_PROMPTS[:N_PROMPTS]

# The authentic arena verdict: show the judge both answers and ask which is
# better. We present each pair in BOTH orders and keep only order-consistent
# verdicts — the standard position-bias correction. Project 54 measured why this
# matters: a small pairwise judge picks slot A ~98% of the time, so most matchups
# are inconsistent and get discarded, which is itself part of this project's story.
JUDGE_TMPL = """Two AI assistants answered the same question. Pick the better answer.

Question: {q}

Answer A: {a}

Answer B: {b}

Which answer is more helpful and correct? Reply with exactly one letter, A or B.
Answer:"""


def generate_all():
    """Model name -> list of answers (one per prompt). Cached."""
    cache = os.path.join(CKPT, "answers.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    answers = {}
    for label, name in CONTESTANTS:
        t = time.time()
        m = Model(name)
        answers[label] = m.answer(PROMPTS, max_new_tokens=MAX_NEW, batch_size=8)
        print(f"  {label:14s} generated {len(PROMPTS)} answers ({time.time()-t:.0f}s)")
        del m
    json.dump(answers, open(cache, "w"))
    return answers


def play_matches(judge, answers):
    """Every pair on every prompt, judged in both orders; keep consistent verdicts.

    Returns wins[i,j] = # times i beat j, the (winner, loser) match list for Elo,
    and the count discarded because the verdict flipped when we swapped the slots
    (i.e. the judge was voting for a position, not an answer).
    """
    labels = [l for l, _ in CONTESTANTS]
    prompts_fwd, prompts_rev, meta = [], [], []
    for i, j in itertools.combinations(range(len(labels)), 2):
        for p, q in enumerate(PROMPTS):
            ai, aj = answers[labels[i]][p], answers[labels[j]][p]
            prompts_fwd.append(JUDGE_TMPL.format(q=q, a=ai, b=aj))  # i=A, j=B
            prompts_rev.append(JUDGE_TMPL.format(q=q, a=aj, b=ai))  # j=A, i=B
            meta.append((i, j))
    fwd, _ = judge.judge_choice(prompts_fwd, [" A", " B"], batch_size=16)
    rev, _ = judge.judge_choice(prompts_rev, [" A", " B"], batch_size=16)

    wins = np.zeros((len(labels), len(labels)))
    discarded = 0
    matches = []
    for (i, j), f, r in zip(meta, fwd, rev):
        i_wins_fwd = f == 0   # picked A(=i)
        i_wins_rev = r == 1   # picked B(=i)
        if i_wins_fwd and i_wins_rev:
            wins[i, j] += 1; matches.append((i, j))
        elif (not i_wins_fwd) and (not i_wins_rev):
            wins[j, i] += 1; matches.append((j, i))
        else:
            discarded += 1    # order-inconsistent -> position bias
    return wins, matches, discarded


def elo_ratings(matches, order, k=32, base=1000.0):
    """Online Elo over `matches` processed in the given `order`."""
    n = len(CONTESTANTS)
    r = np.full(n, base)
    for idx in order:
        w, l = matches[idx]
        ew = 1.0 / (1.0 + 10 ** ((r[l] - r[w]) / 400))
        r[w] += k * (1 - ew)
        r[l] += k * (0 - (1 - ew))
    return r


def bootstrap_elo(matches, n_boot=300, seed=0):
    """Bootstrap CI by resampling the *games* with replacement.

    Permuting the processing order alone would only expose Elo's order-dependence,
    not the real question — "how much would the ranking move if we'd sampled a
    different set of matchups?" Resampling games with replacement answers that,
    and with only ~30 order-consistent games it produces honestly wide intervals.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n_boot, len(CONTESTANTS)))
    m = len(matches)
    for b in range(n_boot):
        idx = rng.integers(0, m, m)              # resample games with replacement
        order = rng.permutation(m)               # and a random processing order
        out[b] = elo_ratings([matches[i] for i in idx], order)
    return out


def main():
    t0 = time.time()
    labels = [l for l, _ in CONTESTANTS]
    print("generating answers ...")
    answers = generate_all()

    print(f"\njudge A: {JUDGE_A}")
    ja = Model(JUDGE_A)
    wins_a, matches_a, disc_a = play_matches(ja, answers)
    n_games = len(list(itertools.combinations(range(len(labels)), 2))) * len(PROMPTS)
    print(f"  {n_games} matchups, {len(matches_a)} order-consistent, "
          f"{disc_a} discarded to position bias ({disc_a/n_games:.0%})")

    boot = bootstrap_elo(matches_a)
    mean_elo = boot.mean(0)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    win_rate = wins_a.sum(1) / np.maximum(wins_a.sum(1) + wins_a.sum(0), 1)

    order = np.argsort(-mean_elo)
    print("\n  rank  model           Elo   95% CI            win-rate")
    for rank, k in enumerate(order, 1):
        print(f"  {rank:>2}.   {labels[k]:14s} {mean_elo[k]:6.0f}  "
              f"[{lo[k]:.0f},{hi[k]:.0f}]   {win_rate[k]:.2f}")
    # how many of the top models are a statistical tie (CIs overlap the leader)?
    leader = order[0]
    tied = [k for k in order if hi[k] >= lo[leader]]
    print(f"  --> top {len(tied)} models are a statistical tie "
          f"(CIs overlap the leader's)")

    # --- judge B: does the ranking survive a different judge? -------------- #
    print(f"\njudge B: {JUDGE_B}")
    jb = Model(JUDGE_B)
    wins_b, matches_b, disc_b = play_matches(jb, answers)
    print(f"  {len(matches_b)} order-consistent, {disc_b} discarded to position "
          f"bias ({disc_b/n_games:.0%})")
    usable_b = len(matches_b) >= 20
    if usable_b:
        elo_b = bootstrap_elo(matches_b, n_boot=200, seed=1).mean(0)
        order_b = np.argsort(-elo_b)
        rank_a = {labels[k]: r for r, k in enumerate(order)}
        rank_b = {labels[k]: r for r, k in enumerate(order_b)}
        ra = np.array([rank_a[l] for l in labels])
        rb = np.array([rank_b[l] for l in labels])
        spearman = float(np.corrcoef(ra, rb)[0, 1])
        print(f"  judge B order: {' > '.join(labels[k] for k in order_b)}")
        print(f"  Spearman rank correlation A vs B: {spearman:.3f}")
    else:
        elo_b = np.full(len(labels), np.nan)
        spearman = float("nan")
        print(f"  too few consistent games to rank — this judge is too "
              f"position-biased to run the arena at all")

    with open(os.path.join(OUT, "arena.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "elo_judgeA", "ci_lo", "ci_hi", "win_rate", "elo_judgeB"])
        for k in order:
            eb = "" if np.isnan(elo_b[k]) else f"{elo_b[k]:.0f}"
            w.writerow([labels[k], f"{mean_elo[k]:.0f}", f"{lo[k]:.0f}",
                        f"{hi[k]:.0f}", f"{win_rate[k]:.3f}", eb])
        w.writerow([])
        w.writerow(["spearman_A_vs_B", "" if np.isnan(spearman) else f"{spearman:.3f}"])
        w.writerow(["judgeA_consistent_games", f"{len(matches_a)}"])
        w.writerow(["judgeA_position_bias_discard", f"{disc_a/n_games:.3f}"])
        w.writerow(["judgeB_consistent_games", f"{len(matches_b)}"])

    # ---- figure: Elo with bootstrap CI ----------------------------------- #
    fig, ax = ps.new_axes(8.2, 4.6)
    y = np.arange(len(order))[::-1]
    for yi, k in zip(y, order):
        col = ps.SERIES[0] if k in tied else ps.INK_SECONDARY
        ax.plot([lo[k], hi[k]], [yi, yi], color=col, lw=7, alpha=0.28,
                solid_capstyle="round", zorder=2)
        ax.plot(mean_elo[k], yi, "o", color=col, ms=10, zorder=4)
        if usable_b and not np.isnan(elo_b[k]):
            ax.plot(elo_b[k], yi, "D", color=ps.SERIES[2], ms=6, zorder=5,
                    label="judge B (SmolLM)" if yi == y[0] else None)
        ax.text(hi[k] + 8, yi, f"{mean_elo[k]:.0f}", va="center", fontsize=9,
                color=ps.INK)
    ax.axvspan(lo[leader], max(hi[k] for k in tied), color=ps.SERIES[0], alpha=0.06,
               zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in order])
    ax.axvline(1000, color=ps.BASELINE, lw=1, ls=":", zorder=1)
    ax.set_xlabel("Elo rating (start 1000; bar = 95% CI, bootstrap over resampled games)")
    ax.set_title(f"5-model arena: the top {len(tied)} are a statistical tie "
                 f"({disc_a/n_games:.0%} of games lost to position bias)",
                 color=ps.INK, fontsize=11.5, loc="left")
    if usable_b:
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "arena_elo.png"))
    print(f"\ndone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
