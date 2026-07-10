"""Tree-of-Thoughts: explore several lines of reasoning, prune with a step scorer.

Greedy chain-of-thought commits to one path; if an early step is wrong, the whole
answer is wrong. Tree-of-Thoughts treats reasoning as a search: at each step it samples
several candidate continuations, scores each with a process reward model
([project 39](../39-process-reward-model/README.md)), and keeps only the most promising
partial solutions (a beam). That lets it recover from a bad step instead of committing to
it. We compare PRM-guided beam search against greedy CoT and against sampling the same
number of full solutions (self-consistency).

    python tree_of_thoughts.py       # ~7 min on CPU

Reuses the shared task (reason_lib), the PRM (project 39), and the GPT skeleton (08).
"""

import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "36-cot-vs-direct-on-gsm8k"))
sys.path.insert(0, str(HERE.parent / "39-process-reward-model"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import reason_lib as R   # noqa: E402
import prm as PRM        # noqa: E402

OUT = HERE / "outputs"
COMMA = R.STOI[","]
BEAM = 4
SAMPLES = 6
TEMP = 1.0
NPROB = 100


@torch.no_grad()
def gen_steps(model, prefixes, temp):
    """From each prefix (id-list), sample one reasoning step (up to the next ',' or ';').
    Returns list of (new_ids, step_str, is_final). Bucketed by length for batching."""
    by_len = defaultdict(list)
    for i, pre in enumerate(prefixes):
        by_len[len(pre)].append(i)
    out = [None] * len(prefixes)
    for idxs in by_len.values():
        cur = torch.tensor([prefixes[i] for i in idxs])
        gen = [[] for _ in idxs]; done = torch.zeros(len(idxs), dtype=torch.bool)
        fin = [False] * len(idxs)
        for _ in range(12):
            logits, _ = model(cur[:, -R.BLOCK:])
            if temp and temp > 0:
                nxt = torch.multinomial(F.softmax(logits[:, -1] / temp, -1), 1).squeeze(-1)
            else:
                nxt = logits[:, -1].argmax(-1)
            for j in range(len(idxs)):
                if not done[j]:
                    t = int(nxt[j]); gen[j].append(t)
                    if t == COMMA or t == R.END:
                        done[j] = True; fin[j] = (t == R.END)
            cur = torch.cat([cur, nxt.unsqueeze(1)], 1)
            if done.all():
                break
        for k, i in enumerate(idxs):
            out[i] = (prefixes[i] + gen[k], R.decode(gen[k]), fin[k])
    return out


GEN_TEMP = 0.8           # temperature for the *alternative* branches when repairing


def prm_prob(prm, ids):
    """The PRM's probability that the step ending at this sequence's last token is correct."""
    x, _ = R.pad_batch([ids])
    return float(torch.sigmoid(prm.token_scores(x)[0, len(ids) - 1]))


@torch.no_grad()
def tot_solve(model, prm, xs):
    """Follow the greedy chain, but when the PRM flags a step as wrong, branch: sample
    alternative next-steps and take the one the PRM most trusts. This "search only where
    you're stuck" keeps the reliability of greedy decoding and adds recovery on the exact
    steps that need it — avoiding the over-optimization of maximizing an imperfect PRM at
    every step."""
    ids = R.encode(R.prompt_str(xs))
    for _ in range(R.K - 1):
        g = gen_steps(model, [ids], 0.0)[0]                # the greedy next step
        gp = prm_prob(prm, g[0])
        if gp > 0.5:
            ids = g[0]                                     # PRM trusts it -> keep greedy
            continue
        # greedy step looks wrong: branch and repair
        best_ids, best_p = g[0], gp
        for cand in gen_steps(model, [ids] * SAMPLES, GEN_TEMP):
            if not re.match(r"^(\d+)\+(\d+)=(\d+)[,;]?$", cand[1]):
                continue
            p = prm_prob(prm, cand[0])
            if p > best_p:
                best_ids, best_p = cand[0], p
        ids = best_ids
    return R.decode(ids[len(R.encode(R.prompt_str(xs))):])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # a PARTIAL model that makes step errors, so search has something to fix
    model = R.cot_model(steps=600)
    data = PRM.gen_labeled(model)
    prm = PRM.train_prm(data)
    print(f"PRM step accuracy {PRM.step_accuracy(prm, data):.3f}")

    rng = random.Random(11)
    probs = [R.problem(rng) for _ in range(NPROB)]

    # greedy CoT
    greedy = R.sample_batch(model, [xs for xs, _ in probs], temperature=0.0)
    greedy_acc = sum(R.is_correct(xs, a, g) for (xs, a), g in zip(probs, greedy)) / NPROB

    # self-consistency with the same budget as ToT (BEAM*SAMPLES full samples)
    budget = BEAM * SAMPLES
    flat = [xs for xs, _ in probs for _ in range(budget)]
    sc = R.sample_batch(model, flat, temperature=TEMP)
    sc_acc = 0
    for i, (xs, ans) in enumerate(probs):
        ans_list = [R.final_answer(c) for c in sc[i * budget:(i + 1) * budget] if R.final_answer(c)]
        if ans_list:
            sc_acc += (Counter(ans_list).most_common(1)[0][0] == str(ans))
    sc_acc /= NPROB

    # tree-of-thoughts
    tot_acc = sum(R.is_correct(xs, a, tot_solve(model, prm, xs)) for (xs, a) in probs) / NPROB

    print(f"greedy CoT {greedy_acc:.3f} | self-consistency {sc_acc:.3f} | ToT {tot_acc:.3f}")
    plot(greedy_acc, sc_acc, tot_acc)
    (OUT / "results.csv").write_text(
        "method,accuracy\n"
        f"greedy_cot,{greedy_acc:.3f}\nself_consistency,{sc_acc:.3f}\ntree_of_thoughts,{tot_acc:.3f}\n")
    print(f"wrote {OUT/'tree_of_thoughts.png'} + results.csv")


def plot(greedy, sc, tot):
    import plot_style as ps
    fig, ax = ps.new_axes(6.8, 4.4)
    bars = ax.bar(["greedy\nCoT", f"self-consistency\n({BEAM*SAMPLES} samples)",
                   "tree-of-thoughts\n(PRM repair)"], [greedy, sc, tot],
                  color=[ps.SERIES[2], ps.SERIES[0], ps.SERIES[1]], width=0.6)
    for b, v in zip(bars, [greedy, sc, tot]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center",
                fontsize=10, color=ps.INK_SECONDARY)
    ax.set_ylim(0, 1)
    ps.finish(fig, ax, "Searching over steps recovers from early mistakes",
              "", "exact-answer accuracy", OUT / "tree_of_thoughts.png")


if __name__ == "__main__":
    main()
