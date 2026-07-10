"""Self-consistency: solve the problem many times, trust the answer that recurs.

A single sampled chain-of-thought can go wrong by chance. Self-consistency samples
many independent CoT solutions to the *same* problem and takes a majority vote on the
final answer — random mistakes cancel, and accuracy climbs with the number of samples.
We sweep n in {1, 4, 16, 64} and plot accuracy against that inference cost.

    python self_consistency.py       # ~5 min on CPU

Reuses the shared task (reason_lib) and the GPT skeleton from project 08.
"""

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "36-cot-vs-direct-on-gsm8k"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import reason_lib as R   # noqa: E402

OUT = HERE / "outputs"
NS = [1, 4, 16, 64]
NPROB = 120
NMAX = 64
TEMP = 1.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = R.cot_model(steps=800)
    greedy = R.accuracy(model, n=NPROB, temperature=0.0)
    print(f"greedy CoT accuracy {greedy:.3f}")

    rng = random.Random(0)
    probs = [R.problem(rng) for _ in range(NPROB)]
    # sample NMAX completions per problem ONCE (replicate prompts), reuse as prefixes
    flat = [xs for xs, _ in probs for _ in range(NMAX)]
    comps = R.sample_batch(model, flat, temperature=TEMP)
    per_prob = [comps[i * NMAX:(i + 1) * NMAX] for i in range(NPROB)]

    accs = []
    for n in NS:
        correct = 0
        for (xs, ans), cs in zip(probs, per_prob):
            answers = [R.final_answer(c) for c in cs[:n] if R.final_answer(c)]
            if answers:
                vote = Counter(answers).most_common(1)[0][0]
                correct += (vote == str(ans))
        accs.append(correct / NPROB)
        print(f"  n={n:2d}: self-consistency accuracy {accs[-1]:.3f}")

    plot(greedy, accs)
    with open(OUT / "results.csv", "w") as f:
        f.write("n_samples,accuracy\n")
        for n, a in zip(NS, accs):
            f.write(f"{n},{a:.3f}\n")
    print(f"wrote {OUT/'self_consistency.png'} + results.csv")


def plot(greedy, accs):
    import plot_style as ps
    fig, ax = ps.new_axes(7.2, 4.4)
    ax.plot(NS, accs, "-o", color=ps.SERIES[1], label="self-consistency (majority vote)")
    ax.axhline(greedy, color=ps.BASELINE, ls="--", lw=1)
    ax.text(NS[0], greedy + 0.01, "greedy (1 path)", color=ps.INK_MUTED, fontsize=8)
    ax.set_xscale("log", base=2); ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "More samples, better answers — for a price",
              "number of sampled CoT solutions (log scale = inference cost)",
              "exact-answer accuracy", OUT / "self_consistency.png")


if __name__ == "__main__":
    main()
