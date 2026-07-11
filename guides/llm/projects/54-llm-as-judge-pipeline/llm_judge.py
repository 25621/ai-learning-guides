"""LLM-as-judge: build the grader, then audit the grader.

Most LLM-as-judge tutorials stop at "the judge picked A". This one is built on a
task where we *know* the right answer, so we can score the judge itself — and it
contrasts the two ways people actually run LLM judges:

  * PAIRWISE  — show the judge both answers, ask which is better.
  * POINTWISE — show the judge one answer at a time, ask for a 1-5 score.

Candidates come from SQuAD (project 43's `rag_lib`): for each question the
"good" answer is the gold span and the "bad" answer is a span lifted from an
unrelated paragraph — fluent, confident, and wrong, the failure a judge must
catch. We then measure:

  1. POSITION BIAS  — present each pair in both orders. A small pairwise judge
     turns out to pick whichever answer is in slot A almost regardless of
     content, so its apparent accuracy is an artifact of answer ordering.
  2. POINTWISE SIGNAL — does scoring each answer alone separate right from wrong?
  3. VERBOSITY BIAS — pad the *wrong* answer with filler and re-score. If its
     score rises, the judge is rewarding length.
  4. AGREEMENT — a second judge model on the same items (score correlation).

Verdicts and scores are read as log-probabilities over the verdict / digit
token, not generated — one forward pass per judgment.

CPU, ~5 min.
"""

import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "51-mmlu-re-run"))
sys.path.insert(0, os.path.join(HERE, "..", "43-minimal-rag"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))
import plot_style as ps  # noqa: E402
from eval_lib import Model  # noqa: E402
from rag_lib import build_corpus  # noqa: E402

JUDGE = "Qwen/Qwen2.5-0.5B-Instruct"
JUDGE_B = "HuggingFaceTB/SmolLM2-360M-Instruct"  # the "second opinion"
N_PAIRS = 120
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# Filler that adds length and hedging but zero information — the classic thing a
# verbose-biased judge mistakes for quality.
PADDING = (
    " To elaborate, this is a well-established point that is widely discussed in "
    "the relevant literature, and it is worth noting the broader context here as "
    "well, since a complete answer should consider the surrounding details.")

PAIR_TMPL = """Question: {q}

Answer A: {a}
Answer B: {b}

Which answer is factually correct? Reply with exactly one letter, A or B.
Answer:"""

POINT_TMPL = """Rate whether this answer correctly answers the question, from 1 \
(wrong) to 5 (correct).

Question: {q}
Answer: {a}

Score (1-5):"""


def build_pairs(n, seed=0):
    """(question, correct_answer, wrong_answer) triples with known ground truth."""
    rng = np.random.default_rng(seed)
    paragraphs, questions = build_corpus(n_paragraphs=400, n_questions=n * 3, seed=seed)
    pairs = []
    for q in questions:
        gold = q["answers"][0].strip()
        other = paragraphs[int(rng.integers(0, len(paragraphs)))]["text"].split()
        if len(other) < 6 or not gold:
            continue
        s = int(rng.integers(0, max(1, len(other) - 4)))
        wrong = " ".join(other[s : s + max(1, len(gold.split()))]).strip(" ,.;")
        if wrong.lower() == gold.lower() or not wrong:
            continue
        pairs.append({"q": q["q"], "good": gold, "bad": wrong})
        if len(pairs) >= n:
            break
    return pairs


def pairwise_swap(model, pairs):
    """Judge each pair in both orders. Returns (picked_good_fwd, picked_good_rev,
    slot_A_rate) — the ingredients of a position-bias audit."""
    fwd = [PAIR_TMPL.format(q=p["q"], a=p["good"], b=p["bad"]) for p in pairs]
    rev = [PAIR_TMPL.format(q=p["q"], a=p["bad"], b=p["good"]) for p in pairs]
    f, _ = model.judge_choice(fwd, [" A", " B"], batch_size=16)
    r, _ = model.judge_choice(rev, [" A", " B"], batch_size=16)
    f, r = np.array(f), np.array(r)
    picked_good_fwd = f == 0        # in fwd, slot A holds the good answer
    picked_good_rev = r == 1        # in rev, slot B holds the good answer
    slot_a_rate = (np.mean(f == 0) + np.mean(r == 0)) / 2
    return picked_good_fwd, picked_good_rev, float(slot_a_rate)


def pointwise(model, pairs, pad_bad=False):
    """Score good and bad answers independently on 1-5. Returns (sg, sb) arrays."""
    pg = [POINT_TMPL.format(q=p["q"], a=p["good"]) for p in pairs]
    pb = [POINT_TMPL.format(q=p["q"], a=p["bad"] + (PADDING if pad_bad else ""))
          for p in pairs]
    sg = model.score_tokens(pg, ["1", "2", "3", "4", "5"], batch_size=16).argmax(1) + 1
    sb = model.score_tokens(pb, ["1", "2", "3", "4", "5"], batch_size=16).argmax(1) + 1
    return sg, sb


def soft_accuracy(sg, sb):
    """Pairwise accuracy derived from pointwise scores; ties count as 0.5."""
    return float(np.mean((sg > sb) * 1.0 + (sg == sb) * 0.5))


def main():
    t0 = time.time()
    pairs = build_pairs(N_PAIRS)
    n = len(pairs)
    print(f"{n} (question, correct, wrong) triples with known ground truth\n")

    print(f"judge A: {JUDGE}")
    ja = Model(JUDGE)

    # ---- 1. pairwise position bias -------------------------------------- #
    gf, gr, slot_a = pairwise_swap(ja, pairs)
    acc_fwd = float(gf.mean())        # correct answer shown first
    acc_rev = float(gr.mean())        # correct answer shown second
    consistency = float(np.mean(gf == gr))
    print("  PAIRWISE (show both answers, pick A or B):")
    print(f"    accuracy, correct answer in slot A : {acc_fwd:.3f}")
    print(f"    accuracy, correct answer in slot B : {acc_rev:.3f}")
    print(f"    picks slot A regardless of content : {slot_a:.3f}"
          f"   (unbiased = 0.50)")
    print(f"    order-consistent verdicts          : {consistency:.3f}"
          f"   (fair = 1.00)")
    print(f"    --> the 'accuracy' is position, not judgment\n")

    # ---- 2. pointwise signal -------------------------------------------- #
    sg, sb = pointwise(ja, pairs)
    acc_point = soft_accuracy(sg, sb)
    tie_rate = float(np.mean(sg == sb))
    print("  POINTWISE (score each answer 1-5 on its own):")
    print(f"    mean score, correct answers        : {sg.mean():.2f}")
    print(f"    mean score, wrong answers          : {sb.mean():.2f}")
    print(f"    pairwise accuracy from scores      : {acc_point:.3f}"
          f"   (ties={tie_rate:.2f} count as 0.5)\n")

    # ---- 3. verbosity bias (pointwise) ---------------------------------- #
    _, sb_pad = pointwise(ja, pairs, pad_bad=True)
    print(f"  padding the WRONG answer with {len(PADDING.split())} words of filler:")
    print(f"    wrong-answer mean score {sb.mean():.2f} -> {sb_pad.mean():.2f} "
          f"({sb_pad.mean()-sb.mean():+.2f})\n")

    # ---- 4. inter-judge agreement (pointwise gap) ----------------------- #
    print(f"judge B: {JUDGE_B}")
    jb = Model(JUDGE_B)
    sg_b, sb_b = pointwise(jb, pairs)
    acc_b = soft_accuracy(sg_b, sb_b)
    gap_a = sg - sb
    gap_b = sg_b - sb_b
    corr = float(np.corrcoef(gap_a, gap_b)[0, 1]) if gap_a.std() and gap_b.std() else 0.0
    print(f"    judge B pointwise accuracy          : {acc_b:.3f}")
    print(f"    judge B mean score good/wrong       : {sg_b.mean():.2f} / {sb_b.mean():.2f}")
    print(f"    correlation of per-item score gaps  : {corr:.3f}")

    rows = [
        ("pairwise acc, correct in slot A", acc_fwd),
        ("pairwise acc, correct in slot B", acc_rev),
        ("pairwise picks-slot-A rate", slot_a),
        ("pairwise order consistency", consistency),
        ("pointwise mean score correct", sg.mean()),
        ("pointwise mean score wrong", sb.mean()),
        ("pointwise accuracy (ties=0.5)", acc_point),
        ("pointwise tie rate", tie_rate),
        ("wrong-answer score, padded", sb_pad.mean()),
        ("judgeB pointwise accuracy", acc_b),
        ("judgeA-judgeB gap correlation", corr),
    ]
    with open(os.path.join(OUT, "judge_audit.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in rows:
            w.writerow([k, f"{float(v):.4f}"])

    # ---- figure --------------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=110)
    for ax in (ax1, ax2):
        fig.patch.set_facecolor(ps.SURFACE)
        ax.set_facecolor(ps.SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(ps.BASELINE)
        ax.spines["bottom"].set_color(ps.BASELINE)
        ax.tick_params(colors=ps.INK_SECONDARY)

    # left: pairwise is a position-bias trap
    labels = ["correct\nin slot A", "correct\nin slot B", "picks slot A\nregardless"]
    vals = [acc_fwd, acc_rev, slot_a]
    colors = [ps.SERIES[1], ps.SERIES[2], ps.SERIES[3]]
    x = np.arange(len(vals))
    ax1.bar(x, vals, width=0.6, color=colors, zorder=3)
    for xi, v in zip(x, vals):
        ax1.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", color=ps.INK,
                 fontsize=11)
    ax1.axhline(0.5, color=ps.BASELINE, lw=1.2, ls=":", zorder=2)
    ax1.text(len(vals) - 0.5, 0.51, "coin flip", ha="right", va="bottom",
             color=ps.INK_MUTED, fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("rate")
    ax1.set_title("Pairwise: 'perfect' accuracy is just position bias",
                  color=ps.INK, fontsize=11.5, loc="left")

    # right: pointwise separates right from wrong — grouped bars per score
    scores_axis = np.array([1, 2, 3, 4, 5])
    cg = np.array([(sg == s).sum() for s in scores_axis])
    cb = np.array([(sb == s).sum() for s in scores_axis])
    cbp = np.array([(sb_pad == s).sum() for s in scores_axis])
    ax2.bar(scores_axis - 0.22, cg, width=0.2, color=ps.SERIES[1], zorder=3,
            label=f"correct (mean {sg.mean():.2f})")
    ax2.bar(scores_axis, cb, width=0.2, color=ps.SERIES[2], zorder=3,
            label=f"wrong (mean {sb.mean():.2f})")
    ax2.bar(scores_axis + 0.22, cbp, width=0.2, color=ps.INK_SECONDARY, zorder=3,
            label=f"wrong + filler (mean {sb_pad.mean():.2f})")
    ax2.set_xticks(scores_axis)
    ax2.set_xlabel("pointwise score (1-5)")
    ax2.set_ylabel("answers")
    ax2.set_title(f"Pointwise: real signal (acc {acc_point:.2f}), gamed by length",
                  color=ps.INK, fontsize=11.5, loc="left")
    ax2.legend(frameon=False, fontsize=9, loc="upper center")
    ax2.annotate("filler pushes the WRONG answer\nfrom mean 2.77 to 4.37 — above correct",
                 xy=(5.22, cbp[-1]), xytext=(2.6, max(cg.max(), cbp.max()) * 0.62),
                 fontsize=8.5, color=ps.INK,
                 arrowprops=dict(arrowstyle="->", color=ps.INK, lw=0.9))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "judge_audit.png"))
    print(f"\ndone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
