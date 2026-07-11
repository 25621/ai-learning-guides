"""MMLU re-run: reproduce an open model's published MMLU score yourself.

We evaluate Qwen2.5-0.5B-Instruct on MMLU with the standard cloze protocol
(read the log-probability the model assigns to each answer *letter*), report
accuracy per macro-category, and check the total against the number Qwen's
authors published (~0.45 for this model).

CPU, ~7 min: a stratified sample of `PER_SUBJECT` questions from each of the 57
subjects, scored one forward pass per question.
"""

import csv
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "01-train-a-bpe-from-scratch"),
)
import plot_style as ps  # noqa: E402
from eval_lib import (  # noqa: E402
    MMLU_CATEGORIES, SUBJECT_TO_CATEGORY, Model, load_mmlu, mmlu_cloze, wilson_ci,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# The reference number: Qwen's own report gives MMLU 47.5 for Qwen2.5-0.5B,
# 5-shot, on the *base* model (the Instruct card reports MMLU-Pro instead).
# Our protocol differs — 0-shot, Instruct weights, a 1,254-question sample —
# which is exactly the gap this project exists to make visible.
PUBLISHED = 0.475
PUBLISHED_NOTE = "Qwen2.5-0.5B, 5-shot, base (Qwen2.5 report)"
PER_SUBJECT = 22   # stratified sample per subject -> ~1250 questions
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoints")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)


def main():
    t0 = time.time()
    items = load_mmlu(per_subject=PER_SUBJECT)
    print(f"loaded {len(items)} MMLU questions across {len(MMLU_CATEGORIES)} categories")

    # Scoring is the only expensive step (~7 min). Cache the predictions so the
    # figure can be re-rendered without paying for the model again.
    cache = os.path.join(CKPT, f"preds_{PER_SUBJECT}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            preds = json.load(f)
        print(f"loaded cached predictions ({len(preds)}) — delete {cache} to re-score")
    else:
        print(f"loading {MODEL} ...")
        m = Model(MODEL)
        print(f"  {m.n_params()/1e6:.0f}M params, {time.time()-t0:.1f}s")
        prompts = [mmlu_cloze(it["question"], it["choices"]) for it in items]
        t1 = time.time()
        preds, _ = m.mc_score(prompts, n_choices=4, batch_size=32)
        print(f"scored in {time.time()-t1:.1f}s")
        with open(cache, "w") as f:
            json.dump(preds, f)

    correct = np.array([int(p == it["answer"]) for p, it in zip(preds, items)])

    # per subject then per category
    subj_hits, subj_tot = {}, {}
    for it, c in zip(items, correct):
        subj_hits[it["subject"]] = subj_hits.get(it["subject"], 0) + int(c)
        subj_tot[it["subject"]] = subj_tot.get(it["subject"], 0) + 1

    cat_rows = []
    for cat, subs in MMLU_CATEGORIES.items():
        k = sum(subj_hits[s] for s in subs)
        n = sum(subj_tot[s] for s in subs)
        p, lo, hi = wilson_ci(k, n)
        cat_rows.append((cat, k, n, p, lo, hi))
        print(f"  {cat:16s} {p:.3f}  [{lo:.3f},{hi:.3f}]  (n={n})")

    ok, olo, ohi = wilson_ci(int(correct.sum()), len(correct))
    print(f"\nOVERALL {ok:.3f}  95% CI [{olo:.3f},{ohi:.3f}]  (n={len(correct)})")
    print(f"published {PUBLISHED:.3f}  ({PUBLISHED_NOTE})")
    print(f"  -> {'lands within our CI' if olo <= PUBLISHED <= ohi else 'OUTSIDE our CI'}"
          f"; gap {ok - PUBLISHED:+.3f}")
    print("  note: we ran 0-shot on the Instruct weights, they ran 5-shot on the "
          "base model.\n  Agreement this close is partly luck — project 52 shows the "
          "prompt alone moves\n  this number by several points.")

    # ---- save per-subject CSV ---- #
    with open(os.path.join(OUT, "mmlu_by_subject.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "category", "correct", "n", "accuracy"])
        for s in sorted(subj_tot):
            w.writerow([s, SUBJECT_TO_CATEGORY[s], subj_hits[s], subj_tot[s],
                        f"{subj_hits[s]/subj_tot[s]:.4f}"])

    # ---- figure: per-category accuracy + overall vs published ---- #
    fig, ax = ps.new_axes(7.6, 4.4)
    cats = [r[0] for r in cat_rows]
    accs = [r[3] for r in cat_rows]
    los = [r[3] - r[4] for r in cat_rows]
    his = [r[5] - r[3] for r in cat_rows]
    x = np.arange(len(cats))
    ax.bar(x, accs, width=0.62, color=ps.SERIES[0], zorder=3)
    ax.errorbar(x, accs, yerr=[los, his], fmt="none", ecolor=ps.INK_SECONDARY,
                capsize=4, lw=1.2, zorder=4)
    for xi, a in zip(x, accs):
        ax.text(xi, a + 0.02, f"{a:.2f}", ha="center", va="bottom",
                color=ps.INK, fontsize=10)
    ax.axhline(0.25, color=ps.BASELINE, lw=1.2, ls=":", zorder=2)
    ax.text(len(cats) - 0.5, 0.255, "chance 0.25", ha="right", va="bottom",
            color=ps.INK_MUTED, fontsize=9)
    ax.axhline(ok, color=ps.SERIES[2], lw=1.4, ls="--", zorder=2)
    ax.text(1.5, ok + 0.045, f"our 0-shot overall {ok:.3f}", ha="center",
            va="bottom", color=ps.SERIES[2], fontsize=9)
    ax.annotate("", xy=(1.5, ok + 0.004), xytext=(1.5, ok + 0.045),
                arrowprops=dict(arrowstyle="-", color=ps.SERIES[2], lw=0.8))
    ax.axhline(PUBLISHED, color=ps.SERIES[1], lw=1.4, ls="--", zorder=2)
    ax.text(2.5, PUBLISHED - 0.075, f"published {PUBLISHED:.3f} (5-shot, base)",
            ha="center", va="top", color=ps.SERIES[1], fontsize=9)
    ax.annotate("", xy=(2.5, PUBLISHED - 0.004), xytext=(2.5, PUBLISHED - 0.075),
                arrowprops=dict(arrowstyle="-", color=ps.SERIES[1], lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("accuracy")
    ax.set_title(f"MMLU by category — {MODEL.split('/')[-1]}  (n={len(correct)})",
                 color=ps.INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mmlu_by_category.png"))
    print(f"\ndone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
