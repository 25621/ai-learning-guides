"""Length-budget controller: teach the model to think just long enough.

More reasoning tokens usually mean better answers — but they cost time and money, and
past a point they stop helping. This project trains a model to obey an explicit thinking
budget: a leading token says "show at most B reasoning steps", and the model does exactly
that many partial sums before committing to an answer. With a small budget it must finish
the sum in its head (error-prone); with a full budget it writes out every step (reliable).
We then measure accuracy against the budget — the quality-vs-compute curve.

    python budget.py       # ~5 min on CPU

Reuses the GPT skeleton from project 08.
"""

import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT   # noqa: E402

K = 4; LO, HI = 0, 20
CHARS = "0123456789+=,;>"
STOI = {c: i for i, c in enumerate(CHARS)}; ITOS = {i: c for c, i in STOI.items()}
BLOCK = 44; END = STOI[";"]
BUDGETS = [0, 1, 2, 3]
OUT = HERE / "outputs"


def enc(s):
    return [STOI[c] for c in s]


def problem(rng):
    xs = [rng.randint(LO, HI) for _ in range(K)]
    return xs, sum(xs)


def make(xs, ans, b):
    prompt = f"{b}>" + "+".join(map(str, xs)) + "="
    s = xs[0]; steps = []
    for i in range(b):
        steps.append(f"{s}+{xs[i + 1]}={s + xs[i + 1]}"); s += xs[i + 1]
    if b == K - 1:
        comp = ",".join(steps) + ";"                       # last step already ends in the total
    else:
        comp = (",".join(steps) + "," if steps else "") + f"{ans};"
    return prompt, comp


def final_answer(comp):
    nums = re.findall(r"\d+", comp.split(";")[0])
    return nums[-1] if nums else ""


def batch(bs, rng):
    X, Y, M = [], [], []
    for _ in range(bs):
        xs, ans = problem(rng); b = rng.choice(BUDGETS)
        prompt, comp = make(xs, ans, b); s = prompt + comp
        ids = enc(s) + [END] * (BLOCK + 1 - len(s)); eq = s.index("=")
        X.append(ids[:BLOCK]); Y.append(ids[1:BLOCK + 1])
        M.append([1.0 if i >= eq else 0.0 for i in range(BLOCK)])
    return torch.tensor(X), torch.tensor(Y), torch.tensor(M)


@torch.no_grad()
def eval_budget(model, b, n=200, seed=99):
    rng = random.Random(seed); correct = 0; toks = 0
    probs = [problem(rng) for _ in range(n)]
    # bucket by prompt length for batched generation
    from collections import defaultdict
    by = defaultdict(list)
    prompts = [f"{b}>" + "+".join(map(str, xs)) + "=" for xs, _ in probs]
    for i, p in enumerate(prompts):
        by[len(p)].append(i)
    gens = [None] * n
    for idxs in by.values():
        cur = torch.tensor([enc(prompts[i]) for i in idxs])
        g = [[] for _ in idxs]; done = torch.zeros(len(idxs), dtype=torch.bool)
        for _ in range(BLOCK - 6):
            logits, _ = model(cur[:, -BLOCK:]); nxt = logits[:, -1].argmax(-1)
            for j in range(len(idxs)):
                if not done[j]:
                    t = int(nxt[j]); g[j].append(t)
                    if t == END:
                        done[j] = True
            cur = torch.cat([cur, nxt.unsqueeze(1)], 1)
            if done.all():
                break
        for k, i in enumerate(idxs):
            gens[i] = g[k]
    for (xs, ans), gi in zip(probs, gens):
        comp = "".join(ITOS[t] for t in gi)
        correct += (final_answer(comp) == str(ans)); toks += len(gi)
    return correct / n, toks / n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); torch.set_num_threads(12)
    model = GPT(Config(vocab_size=len(CHARS), n_layer=4, n_head=4, n_embd=128, block_size=BLOCK))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    rng = random.Random(0)
    for step in range(1000):
        x, y, m = batch(64, rng); logits, _ = model(x)
        loss = (F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                reduction="none") * m.reshape(-1)).sum() / m.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    accs, toks = [], []
    for b in BUDGETS:
        a, t = eval_budget(model, b)
        accs.append(a); toks.append(t)
        print(f"budget {b}: accuracy {a:.3f} | {t:.1f} tokens/answer")

    plot(accs, toks)
    with open(OUT / "results.csv", "w") as f:
        f.write("budget_steps,accuracy,tokens_per_answer\n")
        for b, a, t in zip(BUDGETS, accs, toks):
            f.write(f"{b},{a:.3f},{t:.1f}\n")
    print(f"wrote {OUT/'budget.png'} + results.csv")


def plot(accs, toks):
    import plot_style as ps
    fig, ax = ps.new_axes(7.2, 4.4)
    ax.plot(toks, accs, "-o", color=ps.SERIES[1])
    for b, a, t in zip(BUDGETS, accs, toks):
        ax.annotate(f"budget {b}", (t, a), textcoords="offset points", xytext=(6, -10),
                    color=ps.INK_SECONDARY, fontsize=8)
    ax.set_ylim(0, 1)
    ps.finish(fig, ax, "Quality vs. thinking budget: more steps, better answers",
              "tokens generated per answer (the thinking budget)", "exact-answer accuracy",
              OUT / "budget.png")


if __name__ == "__main__":
    main()
