"""Prompt sensitivity sweep: one model, one benchmark, five wordings.

We hold Qwen2.5-0.5B-Instruct and a fixed MMLU sample constant and change only
how the question is *phrased* — five prompt templates that a reasonable person
would consider equivalent. The accuracy moves several points between them, which
is the whole lesson: a benchmark number is a property of (model, prompt), not of
the model alone.

CPU, ~4 min: `N` questions x 5 formats, each one forward pass.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "51-mmlu-re-run"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))
import plot_style as ps  # noqa: E402
from eval_lib import Model, load_mmlu, mmlu_cloze, wilson_ci  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FORMATS = [
    ("plain", "Question: / A. / Answer:"),
    ("no_labels", "bare question / A. / Answer:"),
    ("parenthesized", "(A) options / The answer is ("),
    ("verbose", "expert-exam instruction wrapper"),
    ("chat", "Qwen chat template / assistant turn"),
]
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    t0 = time.time()
    m = Model(MODEL)
    # 4 questions/subject * 57 ~= 228, balanced across all categories
    items = load_mmlu(per_subject=4)
    ans = np.array([it["answer"] for it in items])
    print(f"{len(items)} questions, {len(FORMATS)} formats")

    results = []
    for style, desc in FORMATS:
        prompts = [mmlu_cloze(it["question"], it["choices"], style=style) for it in items]
        t1 = time.time()
        preds, _ = m.mc_score(prompts, n_choices=4, batch_size=32)
        acc = float(np.mean(np.array(preds) == ans))
        p, lo, hi = wilson_ci(int((np.array(preds) == ans).sum()), len(ans))
        results.append((style, desc, acc, lo, hi))
        print(f"  {style:14s} acc={acc:.3f} [{lo:.3f},{hi:.3f}]  ({time.time()-t1:.1f}s)")

    accs = [r[2] for r in results]
    spread = max(accs) - min(accs)
    print(f"\nspread = {spread*100:.1f} points  (min {min(accs):.3f} -> max {max(accs):.3f})")

    # save csv
    import csv
    with open(os.path.join(OUT, "prompt_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["format", "description", "accuracy", "ci_lo", "ci_hi"])
        for r in results:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}"])

    # figure: horizontal bars sorted, spread annotated
    order = np.argsort(accs)
    fig, ax = ps.new_axes(7.6, 4.2)
    y = np.arange(len(results))
    accs_s = [accs[i] for i in order]
    labels = [results[i][0] for i in order]
    los = [accs[i] - results[i][3] for i in order]
    his = [results[i][4] - accs[i] for i in order]
    ax.barh(y, accs_s, height=0.6, color=ps.SERIES[0], zorder=3)
    ax.errorbar(accs_s, y, xerr=[los, his], fmt="none", ecolor=ps.INK_SECONDARY,
                capsize=3, lw=1.1, zorder=4)
    for yi, a in zip(y, accs_s):
        ax.text(a + 0.006, yi, f"{a:.3f}", va="center", ha="left",
                color=ps.INK, fontsize=10)
    ax.axvline(0.25, color=ps.BASELINE, lw=1.2, ls=":", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(accs) + 0.08)
    ax.set_xlabel("MMLU accuracy (same model, same questions)")
    ax.set_title(f"Prompt sensitivity — {spread*100:.1f}-point swing from wording alone",
                 color=ps.INK, fontsize=12, loc="left")
    # shaded spread band
    ax.axvspan(min(accs), max(accs), color=ps.SERIES[2], alpha=0.06, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prompt_sweep.png"))
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
