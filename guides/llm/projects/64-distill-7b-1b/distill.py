"""Distillation: train a small student to imitate a big teacher, and count what survives.

The guide's framing is 7B -> 1B. On this box the same experiment runs at
0.8M -> 0.05M parameters, on the verifiable arithmetic task that carries Phase 5
(project 28's `sft_lib`): the prompt is "37+8=" and the completion is "45;". A
verifiable task is the right choice here because "quality retention" stops being
a vibe — it is exact-match accuracy, and the student either gets the sum right or
it does not.

Four ways to make the small model:

  hard labels    ordinary SFT on ground truth. The teacher is never consulted.
  logit KD       match the teacher's full next-token distribution (soft targets).
  sequence KD    SFT on the teacher's own greedy outputs — including its mistakes.
  scarce labels  the case distillation actually exists for: ground truth is
                 expensive and rare, but the teacher will label as much data as
                 you can generate, for free.

Run:  python3 distill.py           (~8 min)
      python3 distill.py --plot    (redraw from outputs/results.json)
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "28-sft-a-1b-base-model"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))

import sft_lib as S  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoints")
torch.set_num_threads(12)

TEACHER = dict(n_layer=4, n_head=4, n_embd=128)     # the expensive asset
STUDENT = dict(n_layer=2, n_head=2, n_embd=32)      # what we can afford to serve
TEACHER_STEPS = 1800                                 # enough to get well past the grok
STUDENT_STEPS = 900
BATCH = 64
KD_TEMP = 1.0                                        # softens the teacher's distribution
SCARCE_N = 128                                       # labelled examples in the scarce regime
EVAL_N = 500


# ------------------------------------------------------------------ the losses
def kd_loss(student_logits, teacher_logits, y, mask, temp=KD_TEMP, alpha=1.0):
    """KL(teacher || student) on the completion tokens, plus optional hard-label CE.

    The temperature is what carries the 'dark knowledge': at temp 1 a confident
    teacher is nearly one-hot and says little beyond the label. Softened, it also
    says *which wrong answers are close* — for 37+8 it puts mass on 45, some on 44
    and 46, none on 12 — and that ranking is a far richer target than a single id.
    """
    t = F.log_softmax(teacher_logits / temp, dim=-1)
    s = F.log_softmax(student_logits / temp, dim=-1)
    kl = F.kl_div(s, t, log_target=True, reduction="none").sum(-1)      # (B, T)
    kl = (kl * mask).sum() / mask.sum().clamp(min=1) * (temp ** 2)
    if alpha >= 1.0:
        return kl
    return alpha * kl + (1 - alpha) * S.masked_ce(student_logits, y, mask)


# ------------------------------------------------------------------- training
def train_hard(model, steps, rng, fixed=None, log=""):
    """Plain SFT. `fixed` = a finite pool of labelled examples (the scarce regime)."""
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)
    for step in range(steps):
        if fixed is None:
            x, y, m = S.sft_batch(BATCH, rng)
        else:
            x, y, m = sample_pool(fixed, BATCH, rng)
        logits, _ = model(x)
        loss = S.masked_ce(logits, y, m)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def train_kd(model, teacher, steps, rng, alpha=1.0, temp=KD_TEMP):
    """Logit KD: fresh problems every step, targets are the teacher's distribution.

    Note what is *not* here: ground-truth answers. The teacher is the only
    supervision — which is exactly the production setting.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)
    for step in range(steps):
        x, y, m = S.sft_batch(BATCH, rng)
        with torch.no_grad():
            t_logits, _ = teacher(x)
        s_logits, _ = model(x)
        loss = kd_loss(s_logits, t_logits, y, m, temp=temp, alpha=alpha)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def teacher_labelled_batch(teacher, bs, rng):
    """The teacher as a label factory: it answers, and its answer becomes the target."""
    probs = [(rng.randint(0, S.MAXN), rng.randint(0, S.MAXN)) for _ in range(bs)]
    gens = S.sample_batch(teacher, probs, temperature=0.0)
    xs, ys, ms = [], [], []
    for (a, b), g in zip(probs, gens):
        ans = g.split(";")[0]
        if not ans.isdigit():
            ans = "0"                                   # teacher babble: keep it, it is data
        s = f"{a}+{b}={ans};"[:S.BLOCK + 1]
        ids = S.encode(s) + [S.END] * (S.BLOCK + 1 - len(s))
        eq = s.index("=")
        xs.append(ids[:S.BLOCK]); ys.append(ids[1:S.BLOCK + 1])
        ms.append([1.0 if i >= eq else 0.0 for i in range(S.BLOCK)])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(ms)


def train_seq_kd(model, teacher, steps, rng):
    """Sequence-level KD: SFT on what the teacher actually generates."""
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)
    for step in range(steps):
        x, y, m = teacher_labelled_batch(teacher, BATCH, rng)
        logits, _ = model(x)
        loss = S.masked_ce(logits, y, m)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


# ---------------------------------------------------------------- scarce data
def make_pool(n, rng):
    """A small, fixed set of *labelled* examples — all the ground truth we can afford."""
    xs, ys, ms = [], [], []
    for _ in range(n):
        a, b, c = S.sample_problem(rng)
        s = f"{a}+{b}={c};"
        ids = S.encode(s) + [S.END] * (S.BLOCK + 1 - len(s))
        eq = s.index("=")
        xs.append(ids[:S.BLOCK]); ys.append(ids[1:S.BLOCK + 1])
        ms.append([1.0 if i >= eq else 0.0 for i in range(S.BLOCK)])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(ms)


def sample_pool(pool, bs, rng):
    x, y, m = pool
    idx = torch.tensor([rng.randrange(x.shape[0]) for _ in range(bs)])
    return x[idx], y[idx], m[idx]


# ------------------------------------------------------------------ measuring
def latency(model, n=40):
    """Decode latency for one completion — the reason we wanted a small model."""
    probs = [(7, 8)] * 8
    model(torch.zeros(1, S.BLOCK, dtype=torch.long))      # warm
    t0 = time.perf_counter()
    for _ in range(n):
        S.sample_batch(model, probs, temperature=0.0)
    return (time.perf_counter() - t0) / (n * len(probs)) * 1000


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def evaluate(name, model, teacher_acc=None):
    acc = S.accuracy(model, n=EVAL_N)
    row = dict(name=name, acc=acc, params=n_params(model), ms=latency(model),
               retention=(acc / teacher_acc if teacher_acc else 1.0))
    print(f"  {name:24s} | acc {acc:.3f} | {row['params']:7,d} params | "
          f"{row['ms']:5.2f} ms/completion"
          + (f" | retains {row['retention']:5.1%} of teacher" if teacher_acc else ""),
          flush=True)
    return row


# -------------------------------------------------------------------- report
def plot(rows):
    teacher = rows[0]
    students = rows[1:]

    # 1. quality retention
    fig, ax = ps.new_axes(7.6, 4.3)
    names = [r["name"] for r in rows]
    colors = [ps.INK_SECONDARY] + [ps.SERIES[2], ps.SERIES[0], ps.SERIES[1],
                                   ps.SERIES[3], ps.SERIES[4]][:len(students)]
    ax.bar(range(len(rows)), [r["acc"] for r in rows], color=colors, width=0.6, zorder=3)
    ax.axhline(teacher["acc"], color=ps.INK_SECONDARY, ls="--", lw=1.2)
    ax.annotate(f"teacher ({teacher['acc']:.0%})", xy=(len(rows) - 0.5, teacher["acc"]),
                xytext=(0, 5), textcoords="offset points", ha="right",
                color=ps.INK_SECONDARY, fontsize=9)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['acc']:.0%}", xy=(i, r["acc"]), xytext=(0, 4),
                    textcoords="offset points", ha="center",
                    color=ps.INK_SECONDARY, fontsize=9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ps.finish(fig, ax, "Quality retention: same student size, same step budget",
              "", "exact-match accuracy", os.path.join(OUT, "retention.png"))

    # 2. the trade the whole phase is about: quality per unit of serving cost
    fig, ax = ps.new_axes(7.2, 4.3)
    for i, r in enumerate(rows):
        c = ps.INK_SECONDARY if i == 0 else colors[i]
        ax.scatter(r["ms"], r["acc"], s=110, color=c, zorder=3)
        ax.annotate(f"{r['name']}\n{r['params']/1000:.0f}k params",
                    xy=(r["ms"], r["acc"]), xytext=(8, -4),
                    textcoords="offset points", color=c, fontsize=8.5)
    ax.set_xlim(0, max(r["ms"] for r in rows) * 1.5)
    ax.set_ylim(0, 1.1)
    ps.finish(fig, ax, "The point of distillation: keep the quality, drop the cost",
              "decode latency (ms per completion)", "exact-match accuracy",
              os.path.join(OUT, "quality_vs_cost.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
    cache = os.path.join(CKPT, "results.json")

    if args.plot:
        plot(json.load(open(cache)))
        return

    rng = random.Random(0)
    print(f"teacher {TEACHER} | student {STUDENT} | {STUDENT_STEPS} steps each\n")

    # -- the teacher: expensive, trained once ------------------------------- #
    tpath = os.path.join(CKPT, "teacher.pt")
    teacher = S.new_model(**TEACHER)
    if os.path.exists(tpath):
        teacher.load_state_dict(torch.load(tpath))
    else:
        print("training the teacher...", flush=True)
        S.train_sft(teacher, steps=TEACHER_STEPS, seed=0)
        torch.save(teacher.state_dict(), tpath)
    teacher.eval()

    rows = [evaluate("teacher (0.8M)", teacher)]
    t_acc = rows[0]["acc"]

    # -- students, all the same size and step budget ------------------------- #
    print("\n[1/2] plenty of labelled data — is the teacher worth consulting?")
    s1 = train_hard(S.new_model(**STUDENT, seed=1), STUDENT_STEPS, random.Random(1))
    rows.append(evaluate("hard labels", s1, t_acc))

    s2 = train_kd(S.new_model(**STUDENT, seed=1), teacher, STUDENT_STEPS,
                  random.Random(1), temp=1.0)
    rows.append(evaluate("logit KD (T=1)", s2, t_acc))

    s2b = train_kd(S.new_model(**STUDENT, seed=1), teacher, STUDENT_STEPS,
                   random.Random(1), temp=2.0)
    rows.append(evaluate("logit KD (T=2)", s2b, t_acc))

    s3 = train_seq_kd(S.new_model(**STUDENT, seed=1), teacher, STUDENT_STEPS,
                      random.Random(1))
    rows.append(evaluate("sequence KD", s3, t_acc))

    # -- the regime distillation is really for ------------------------------- #
    print(f"\n[2/2] now only {SCARCE_N} labelled examples exist. The teacher does not care:")
    print("      it never read a label in the first place, so its student is unchanged.")
    pool = make_pool(SCARCE_N, random.Random(7))
    s4 = train_hard(S.new_model(**STUDENT, seed=1), STUDENT_STEPS, random.Random(1),
                    fixed=pool)
    rows.append(evaluate(f"hard labels (only {SCARCE_N})", s4, t_acc))
    best_kd = max(rows[2:5], key=lambda r: r["acc"])
    print(f"  {'best distilled student':24s} | acc {best_kd['acc']:.3f} | "
          f"({best_kd['name']}) — unchanged, because it never read a label")

    json.dump(rows, open(cache, "w"), indent=1)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    plot(rows)


if __name__ == "__main__":
    main()
