"""Process reward model: grade every step of the reasoning, not just the answer.

An outcome reward model (project 38) gives one score to a whole solution. A process
reward model (PRM) scores *each step* — here, each "A+B=C" line — so it can point at the
exact place the reasoning went wrong. We train a PRM by supervising the reward at every
step boundary with that step's correctness (C == A+B), then use its per-step scores to
*filter* sampled solutions (drop any chain with a step that looks wrong) and vote among
the survivors — beating a plain majority by removing the chains the PRM flags as broken.

    python prm.py       # ~6 min on CPU

Reuses the shared task and RewardModel (reason_lib) and the GPT skeleton from project 08.
"""

import random
import re
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


def step_supervision(xs, completion):
    """Full-sequence ids + list of (boundary_position, step_correct_label)."""
    p = R.prompt_str(xs)
    seq = R.encode(p + completion)
    plen = len(p)
    pos_labels = []
    idx = 0
    for part in completion.split(";")[0].split(","):
        end = idx + len(part)                              # position of the separator after step
        m = re.match(r"^(\d+)\+(\d+)=(\d+)$", part)
        if m and plen + end < len(seq):
            lab = 1.0 if int(m.group(3)) == int(m.group(1)) + int(m.group(2)) else 0.0
            pos_labels.append((plen + end, lab))
        idx = end + 1
    return seq, pos_labels


def gen_labeled(model, n_problems=800, k=4):
    rng = random.Random(2)
    probs = [R.problem(rng) for _ in range(n_problems)]
    flat = [xs for xs, _ in probs for _ in range(k)]
    comps = R.sample_batch(model, flat, temperature=TEMP)
    data = []
    for i, (xs, ans) in enumerate(probs):
        for c in comps[i * k:(i + 1) * k]:
            seq, pl = step_supervision(xs, c)
            if pl:
                data.append((seq, pl))
    return data


def train_prm(data, steps=350, lr=2e-3):
    prm = R.RewardModel()
    opt = torch.optim.AdamW(prm.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = random.Random(0)
    for _ in range(steps):
        batch = [data[rng.randrange(len(data))] for _ in range(48)]
        seqs = [s for s, _ in batch]
        x, _ = R.pad_batch(seqs)
        scores = prm.token_scores(x)                       # (B, T)
        loss = 0.0; count = 0
        for b, (_, pl) in enumerate(batch):
            for pos, lab in pl:
                loss = loss + F.binary_cross_entropy_with_logits(
                    scores[b, pos], torch.tensor(lab)); count += 1
        (loss / count).backward()
        torch.nn.utils.clip_grad_norm_(prm.parameters(), 1.0)
        opt.step(); opt.zero_grad()
    return prm


@torch.no_grad()
def step_accuracy(prm, data):
    correct = tot = 0
    for seq, pl in data[:400]:
        x, _ = R.pad_batch([seq])
        s = prm.token_scores(x)[0]
        for pos, lab in pl:
            correct += ((s[pos] > 0).item() == (lab > 0.5)); tot += 1
    return correct / max(1, tot)


@torch.no_grad()
def prm_solution_score(prm, xs, completion):
    """Weakest-link score: the minimum step probability (min over the reasoning)."""
    seq, pl = step_supervision(xs, completion)
    if not pl:
        return -1e9
    x, _ = R.pad_batch([seq])
    s = torch.sigmoid(prm.token_scores(x)[0])
    return min(float(s[pos]) for pos, _ in pl)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = R.cot_model(steps=800)
    data = gen_labeled(model)
    n_steps = sum(len(pl) for _, pl in data)
    n_wrong = sum(lab < 0.5 for _, pl in data for _, lab in pl)
    print(f"labeled {len(data)} solutions, {n_steps} steps ({100*n_wrong/n_steps:.0f}% wrong)")
    prm = train_prm(data)
    sa = step_accuracy(prm, data)
    print(f"PRM per-step classification accuracy {sa:.3f}")

    # re-ranking: sample 32 per problem, select by PRM weakest-link vs majority
    rng = random.Random(9)
    probs = [R.problem(rng) for _ in range(120)]
    kmax = max(NS)
    flat = [xs for xs, _ in probs for _ in range(kmax)]
    comps = R.sample_batch(model, flat, temperature=TEMP)
    per = [comps[i * kmax:(i + 1) * kmax] for i in range(len(probs))]

    prm_acc, maj_acc = [], []
    for n in NS:
        pc = mc = 0
        for (xs, ans), cs in zip(probs, per):
            cand = cs[:n]
            # PRM-FILTERED majority: drop candidates whose weakest step looks wrong,
            # then vote among the survivors. Filtering (not arg-max selection) uses the
            # scorer robustly — it removes likely-wrong chains instead of chasing the
            # single highest score, which would just find the PRM's blind spots.
            kept = [c for c in cand if prm_solution_score(prm, xs, c) > 0.5]
            pool = kept if kept else cand
            pans = [R.final_answer(c) for c in pool if R.final_answer(c)]
            if pans:
                pc += (Counter(pans).most_common(1)[0][0] == str(ans))
            answers = [R.final_answer(c) for c in cand if R.final_answer(c)]
            if answers:
                mc += (Counter(answers).most_common(1)[0][0] == str(ans))
        prm_acc.append(pc / len(probs)); maj_acc.append(mc / len(probs))
        print(f"  N={n:2d}: PRM-filtered majority {prm_acc[-1]:.3f} | plain majority {maj_acc[-1]:.3f}")

    plot(prm_acc, maj_acc, sa)
    with open(OUT / "results.csv", "w") as f:
        f.write(f"prm_step_classification_accuracy,{sa:.3f}\n")
        f.write("N,prm_rerank_accuracy,majority_accuracy\n")
        for n, p, m in zip(NS, prm_acc, maj_acc):
            f.write(f"{n},{p:.3f},{m:.3f}\n")
    print(f"wrote {OUT/'prm.png'} + results.csv")


def plot(prm_acc, maj_acc, sa):
    import plot_style as ps
    fig, ax = ps.new_axes(7.2, 4.4)
    ax.plot(NS, prm_acc, "-o", color=ps.SERIES[1], label="PRM-filtered majority")
    ax.plot(NS, maj_acc, "-o", color=ps.SERIES[0], label="plain majority")
    ax.set_xscale("log", base=2); ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=9)
    ax.text(NS[0], 0.05, f"PRM step-classification acc {sa:.0%}", color=ps.INK_MUTED, fontsize=8)
    ps.finish(fig, ax, "A process reward model re-ranks by its weakest step",
              "number of candidates N (log scale)", "exact-answer accuracy", OUT / "prm.png")


if __name__ == "__main__":
    main()
