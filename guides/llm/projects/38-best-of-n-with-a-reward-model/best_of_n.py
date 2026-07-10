"""Best-of-N: generate many answers, keep the one a learned judge likes best.

Self-consistency (project 37) trusts the *most common* answer. Best-of-N instead uses a
reward model to judge candidates. We train an outcome reward model to tell correct
solutions from incorrect ones (a binary classifier on verifier-labeled samples), then let
it *filter* the candidates (drop the ones it judges wrong) and vote among the survivors —
a robust use of the scorer that beats a plain vote without over-optimizing against the
reward model's mistakes, which naive arg-max Best-of-N would do as N grows.

    python best_of_n.py       # ~7 min on CPU

Reuses the shared task and RewardModel (reason_lib) and the GPT skeleton from project 08.
"""

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "36-cot-vs-direct-on-gsm8k"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import reason_lib as R   # noqa: E402

OUT = HERE / "outputs"
NS = [1, 4, 16, 32]
TEMP = 1.2


def last_scores(rm, seqs):
    x, lens = R.pad_batch(seqs)
    s = rm.token_scores(x)
    return s[torch.arange(len(seqs)), torch.tensor([l - 1 for l in lens])]


def labeled_pool(model, n_problems=500, k=8):
    """Sample k solutions per problem and label each correct/incorrect by the verifier."""
    rng = random.Random(1)
    probs = [R.problem(rng) for _ in range(n_problems)]
    flat = [xs for xs, _ in probs for _ in range(k)]
    comps = R.sample_batch(model, flat, temperature=TEMP)
    seqs, labels = [], []
    for i, (xs, ans) in enumerate(probs):
        for c in comps[i * k:(i + 1) * k]:
            seqs.append(R.seq_from(xs, c)[0])
            labels.append(1.0 if R.is_correct(xs, ans, c) else 0.0)
    return seqs, torch.tensor(labels)


def train_rm(seqs, labels, steps=400, lr=2e-3):
    rm = R.RewardModel()
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = random.Random(0)
    n = len(seqs)
    for _ in range(steps):
        idx = [rng.randrange(n) for _ in range(64)]
        s = last_scores(rm, [seqs[i] for i in idx])
        loss = F.binary_cross_entropy_with_logits(s, labels[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0); opt.step()
    return rm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = R.cot_model(steps=800)
    greedy = R.accuracy(model, n=150)
    seqs, labels = labeled_pool(model)
    print(f"greedy CoT {greedy:.3f} | reward model on {len(seqs)} labeled solutions "
          f"({labels.mean():.0%} correct)")
    rm = train_rm(seqs, labels)

    # evaluation: sample 32 per problem once, reuse as prefixes for each N
    rng = random.Random(7)
    probs = [R.problem(rng) for _ in range(120)]
    kmax = max(NS)
    flat = [xs for xs, _ in probs for _ in range(kmax)]
    comps = R.sample_batch(model, flat, temperature=TEMP)
    per = [comps[i * kmax:(i + 1) * kmax] for i in range(len(probs))]

    # reward-model discrimination on held-out samples
    disc_c = disc_t = 0
    with torch.no_grad():
        for (xs, ans), cs in zip(probs, per):
            sc = last_scores(rm, [R.seq_from(xs, c)[0] for c in cs])
            for c, s in zip(cs, sc):
                disc_c += ((s > 0).item() == R.is_correct(xs, ans, c)); disc_t += 1
    disc = disc_c / disc_t

    bon_acc, maj_acc = [], []
    for n in NS:
        bon = maj = 0
        for (xs, ans), cs in zip(probs, per):
            cand = cs[:n]
            with torch.no_grad():
                scores = last_scores(rm, [R.seq_from(xs, c)[0] for c in cand])
            # reward-model-FILTERED majority: keep the candidates the RM judges correct,
            # then vote among them. Filtering uses the scorer robustly; taking the single
            # arg-max instead over-optimizes against the RM's mistakes as N grows.
            kept = [c for c, s in zip(cand, scores) if s > 0]
            pool = kept if kept else cand
            pans = [R.final_answer(c) for c in pool if R.final_answer(c)]
            if pans:
                bon += (Counter(pans).most_common(1)[0][0] == str(ans))
            answers = [R.final_answer(c) for c in cand if R.final_answer(c)]
            if answers:
                maj += (Counter(answers).most_common(1)[0][0] == str(ans))
        bon_acc.append(bon / len(probs)); maj_acc.append(maj / len(probs))
        print(f"  N={n:2d}: RM-filtered majority {bon_acc[-1]:.3f} | plain majority {maj_acc[-1]:.3f}")
    print(f"reward-model discrimination accuracy {disc:.3f}")

    plot(bon_acc, maj_acc, disc)
    with open(OUT / "results.csv", "w") as f:
        f.write(f"reward_model_discrimination,{disc:.3f}\n")
        f.write("N,best_of_n_accuracy,majority_accuracy\n")
        for n, b, m in zip(NS, bon_acc, maj_acc):
            f.write(f"{n},{b:.3f},{m:.3f}\n")
    print(f"wrote {OUT/'best_of_n.png'} + results.csv")


def plot(bon, maj, disc):
    import plot_style as ps
    fig, ax = ps.new_axes(7.2, 4.4)
    ax.plot(NS, bon, "-o", color=ps.SERIES[1], label="reward-model-filtered majority")
    ax.plot(NS, maj, "-o", color=ps.SERIES[0], label="plain majority")
    ax.set_xscale("log", base=2); ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=9)
    ax.text(NS[0], 0.05, f"reward-model discrimination {disc:.0%}", color=ps.INK_MUTED, fontsize=8)
    ps.finish(fig, ax, "Best-of-N vs. majority vote at equal samples",
              "number of candidates N (log scale)", "exact-answer accuracy", OUT / "best_of_n.png")


if __name__ == "__main__":
    main()
