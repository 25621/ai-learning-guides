"""Custom eval: a 100-prompt suite shaped like one application, graded 3x.

The application is a customer-support assistant for a fictional online
electronics store, "VoltMart". We write 100 prompts that look like the messages
its real users send — order status, returns, shipping, specs, troubleshooting,
and a few out-of-scope requests it should deflect — and grade every model reply
three times so we can separate two things a single leaderboard number fuses:

  * SIGNAL — real differences between prompts (some are handled well, some badly).
  * NOISE  — the grader's own inconsistency. We grade with three differently
             worded rubrics (the analogue of three human raters) and measure how
             much a prompt's score wobbles between them.

The headline statistic is the intraclass correlation ICC = signal / (signal +
noise): the fraction of the score variance that is real. If ICC is low, your
eval is measuring the grader's mood, not the model.

Model under test: SmolLM2-360M-Instruct (fast generation).
Grader:           Qwen2.5-0.5B-Instruct (scores a 1-5 rubric by log-prob).

CPU, ~4 min.
"""

import csv
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

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
GRADER = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = ("You are VoltMart's customer support assistant. VoltMart is an online "
          "electronics store. Be concise, helpful, and polite.")
OUT = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoints")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)


# --------------------------------------------------------------------------- #
# The 100-prompt eval — programmatically assembled from templates so the mix is
# controlled and reproducible, the way a real application eval is versioned.
# --------------------------------------------------------------------------- #
def build_prompts(seed=0):
    rng = np.random.default_rng(seed)
    items = []            # (category, prompt)
    orders = [f"VM-{rng.integers(10000,99999)}" for _ in range(40)]
    products = ["the AeroBook 14 laptop", "the PulseBuds Pro earbuds",
                "the VoltCharge 65W adapter", "the Nimbus 4K monitor",
                "the ThermoMug smart flask", "the GridCam doorbell"]

    def add(cat, text):
        items.append((cat, text))

    for i in range(16):   # order status
        add("order_status",
            f"Where is my order {orders[i]}? It was supposed to arrive yesterday.")
    for i in range(16):   # returns
        add("returns",
            f"I want to return {products[i % len(products)]}. How do I start a return?")
    for i in range(16):   # shipping
        add("shipping",
            rng.choice([f"Do you ship {products[i % len(products)]} to Canada?",
                        "How long does standard shipping take?",
                        "Can I change the delivery address on an order I just placed?"]))
    for i in range(16):   # product specs
        add("specs",
            f"What's the battery life of {products[i % len(products)]}?")
    for i in range(16):   # troubleshooting
        add("troubleshooting",
            f"{products[i % len(products)]} won't turn on. What should I try?")
    for i in range(12):   # out of scope — should be politely deflected
        add("out_of_scope",
            rng.choice(["Can you write me a poem about my cat?",
                        "What do you think about the upcoming election?",
                        "Help me with my calculus homework.",
                        "What's the meaning of life?"]))
    for i in range(8):    # billing
        add("billing",
            f"I was charged twice for order {orders[20 + i]}. Can you help?")

    rng.shuffle(items)
    return items[:100]


# --------------------------------------------------------------------------- #
# Three rubrics that ask the same question in three ways — the "three raters".
# --------------------------------------------------------------------------- #
RUBRICS = [
    ("plain",
     "Rate how well the support reply answers the customer, from 1 (useless) to "
     "5 (excellent).\n\nCustomer: {q}\nReply: {a}\n\nScore (1-5):"),
    ("criteria",
     "Grade this customer-support reply on helpfulness, correctness, and tone. "
     "1 = poor on all, 5 = strong on all.\n\nCustomer message: {q}\n"
     "Assistant reply: {a}\n\nOverall grade (1-5):"),
    ("strict",
     "You are a strict QA reviewer for a support team. Give 5 only for a reply "
     "that fully resolves the request; give 1 if it is unhelpful or off-topic.\n\n"
     "Question: {q}\nResponse: {a}\n\nRating (1-5):"),
]


def grade(grader, prompts, answers):
    """Return a (n_prompts, 3) integer grade matrix — three rubrics per reply."""
    grades = np.zeros((len(prompts), len(RUBRICS)), dtype=int)
    for r, (_, tmpl) in enumerate(RUBRICS):
        gp = [tmpl.format(q=q, a=a) for (_, q), a in zip(prompts, answers)]
        # score by log-prob over the digit tokens "1".."5"; argmax -> the grade
        probs = grader.score_tokens(gp, ["1", "2", "3", "4", "5"], batch_size=16)
        grades[:, r] = probs.argmax(1) + 1
    return grades


def main():
    t0 = time.time()
    prompts = build_prompts()
    cats = [c for c, _ in prompts]
    print(f"{len(prompts)} prompts across {len(set(cats))} categories")

    ans_cache = os.path.join(CKPT, "answers.json")
    if os.path.exists(ans_cache):
        answers = json.load(open(ans_cache))
        print(f"loaded {len(answers)} cached answers")
    else:
        m = Model(MODEL)
        print(f"model under test: {MODEL} ({m.n_params()/1e6:.0f}M)")
        t = time.time()
        answers = m.answer([q for _, q in prompts], system=SYSTEM,
                           max_new_tokens=48, batch_size=8)
        print(f"generated {len(answers)} replies in {time.time()-t:.0f}s")
        json.dump(answers, open(ans_cache, "w"))

    grader = Model(GRADER)
    print(f"grader: {GRADER}")
    t = time.time()
    G = grade(grader, prompts, answers)   # (100, 3)
    print(f"graded 3x in {time.time()-t:.0f}s\n")

    # --- variance decomposition ------------------------------------------- #
    per_prompt_mean = G.mean(1)                 # (100,)
    grand = G.mean()
    # between-prompt variance (signal): variance of the per-prompt means
    var_between = per_prompt_mean.var(ddof=1)
    # within-prompt variance (noise): mean over prompts of the 3-rater variance
    var_within = G.var(axis=1, ddof=1).mean()
    icc = var_between / (var_between + var_within + 1e-12)
    # how often do all three raters land on the same integer grade?
    unanimous = float(np.mean([len(set(row)) == 1 for row in G]))
    mean_spread = float(np.mean(G.max(1) - G.min(1)))

    print(f"  grand mean grade         {grand:.2f} / 5")
    print(f"  between-prompt variance  {var_between:.3f}   (signal)")
    print(f"  within-prompt variance   {var_within:.3f}   (grader noise)")
    print(f"  ICC (signal fraction)    {icc:.3f}")
    print(f"  3 raters unanimous on    {unanimous:.0%} of prompts "
          f"(mean grade spread {mean_spread:.2f})\n")

    # per-category mean (uses all 3 grades)
    cat_scores = {}
    for c in sorted(set(cats)):
        idx = [i for i, cc in enumerate(cats) if cc == c]
        cat_scores[c] = float(G[idx].mean())
        print(f"    {c:16s} {cat_scores[c]:.2f}  (n={len(idx)})")

    with open(os.path.join(OUT, "custom_eval.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in [("grand_mean", grand), ("var_between_signal", var_between),
                     ("var_within_noise", var_within), ("icc", icc),
                     ("unanimous_frac", unanimous), ("mean_grade_spread", mean_spread)]:
            w.writerow([k, f"{v:.4f}"])
        w.writerow([])
        w.writerow(["category", "mean_grade"])
        for c, v in cat_scores.items():
            w.writerow([c, f"{v:.3f}"])

    # ---- figure ---------------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=110)
    for ax in (ax1, ax2):
        fig.patch.set_facecolor(ps.SURFACE)
        ax.set_facecolor(ps.SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(ps.BASELINE)
        ax.spines["bottom"].set_color(ps.BASELINE)
        ax.tick_params(colors=ps.INK_SECONDARY)

    # left: per-category mean grade
    cs = sorted(cat_scores, key=cat_scores.get)
    y = np.arange(len(cs))
    ax1.barh(y, [cat_scores[c] for c in cs], color=ps.SERIES[0], height=0.62, zorder=3)
    for yi, c in zip(y, cs):
        ax1.text(cat_scores[c] + 0.03, yi, f"{cat_scores[c]:.2f}", va="center",
                 fontsize=9, color=ps.INK)
    ax1.set_yticks(y)
    ax1.set_yticklabels(cs, fontsize=9)
    ax1.set_xlim(1, 5)
    ax1.set_xlabel("mean grade (1-5)")
    ax1.set_title("Where the app is strong and weak", color=ps.INK, fontsize=12,
                  loc="left")

    # right: signal vs noise
    ax2.bar([0, 1], [var_between, var_within], width=0.55,
            color=[ps.SERIES[1], ps.SERIES[2]], zorder=3)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["signal\n(between prompts)", "noise\n(between raters)"],
                        fontsize=10)
    for xi, v in zip([0, 1], [var_between, var_within]):
        ax2.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", color=ps.INK,
                 fontsize=11)
    ax2.set_ylabel("variance of grades")
    ax2.set_title(f"ICC = {icc:.2f}: only {icc:.0%} of the score variance is real",
                  color=ps.INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "custom_eval.png"))
    print(f"\ndone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
