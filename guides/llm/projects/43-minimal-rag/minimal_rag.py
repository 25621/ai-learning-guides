"""Project 43 — Minimal RAG: embed 1,000 Wikipedia paragraphs, retrieve, answer.

Pipeline: MiniLM bi-encoder -> cosine top-k -> extractive reader over the
retrieved paragraphs. Evaluated two ways:

  * retrieval quality — recall@k / nDCG@10 against the gold paragraph
  * end-to-end answers — SQuAD EM/F1, bracketed by two controls:
      "random ctx"  = same reader, 3 random paragraphs (retrieval turned off)
      "oracle ctx"  = same reader, the gold paragraph (perfect retrieval)

Everything between those two brackets is retrieval's contribution.
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))

import plot_style as ps  # noqa: E402
from rag_lib import (Embedder, Reader, build_corpus, dense_search, em_score,  # noqa: E402
                     f1_score, ndcg_at_k, recall_at_k)

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
K_CTX = 3  # paragraphs handed to the reader


def main():
    rng = np.random.default_rng(0)
    paragraphs, questions = build_corpus(n_paragraphs=1000, n_questions=300)
    texts = [p["text"] for p in paragraphs]

    print("== embedding corpus ==", flush=True)
    embedder = Embedder()
    t0 = time.time()
    doc_vecs = embedder.encode(texts, verbose=True)
    print(f"embedded {len(texts)} paragraphs in {time.time() - t0:.1f}s")
    q_vecs = embedder.encode([q["q"] for q in questions])

    print("== retrieval quality ==", flush=True)
    top, _ = dense_search(q_vecs, doc_vecs, k=10)
    ks = [1, 2, 3, 5, 10]
    recalls = {k: np.mean([recall_at_k(top[i], q["gold"], k)
                           for i, q in enumerate(questions)]) for k in ks}
    ndcg = np.mean([ndcg_at_k(top[i], q["gold"], 10)
                    for i, q in enumerate(questions)])
    for k in ks:
        print(f"  recall@{k:<2d} {recalls[k]:.3f}")
    print(f"  nDCG@10   {ndcg:.3f}")

    print("== end-to-end answers ==", flush=True)
    reader = Reader()
    conditions = {"random ctx": None, "RAG top-3": None, "oracle ctx": None}
    rows = []
    for name in conditions:
        t0, em, f1 = time.time(), [], []
        for i, q in enumerate(questions):
            if name == "random ctx":
                ctx_ids = rng.choice(len(texts), size=K_CTX, replace=False)
            elif name == "RAG top-3":
                ctx_ids = top[i][:K_CTX]
            else:
                ctx_ids = [q["gold"]]
            pred, _ = reader.answer(q["q"], [texts[j] for j in ctx_ids])
            em.append(em_score(pred, q["answers"]))
            f1.append(f1_score(pred, q["answers"]))
            if i % 100 == 0:
                print(f"  {name}: {i}/{len(questions)}", flush=True)
        conditions[name] = (np.mean(em), np.mean(f1))
        print(f"  {name:<11s} EM {np.mean(em):.3f}  F1 {np.mean(f1):.3f}  "
              f"({time.time() - t0:.0f}s)")
        rows.append([name, f"{np.mean(em):.3f}", f"{np.mean(f1):.3f}"])

    # a few worked examples for the README
    print("== samples ==")
    for i in [1, 3, 5]:
        q = questions[i]
        pred, _ = reader.answer(q["q"], [texts[j] for j in top[i][:K_CTX]])
        hit = "hit" if q["gold"] in top[i][:K_CTX] else "MISS"
        print(f"  Q: {q['q']}\n     gold={q['answers'][0]!r} "
              f"pred={pred!r} retrieval={hit}")

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "EM", "F1"])
        w.writerows(rows)
        for k in ks:
            w.writerow([f"recall@{k}", f"{recalls[k]:.3f}", ""])
        w.writerow(["nDCG@10", f"{ndcg:.3f}", ""])

    # figure: recall@k curve + EM/F1 bars for the three conditions
    fig, axes = ps.plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ax.set_facecolor(ps.SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ps.BASELINE)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=9)
        ax.grid(True, color=ps.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].plot(ks, [recalls[k] for k in ks], color=ps.SERIES[0],
                 marker="o", linewidth=2)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Gold paragraph found in top-k", color=ps.INK,
                      fontsize=11, loc="left")
    axes[0].set_xlabel("k (paragraphs retrieved)", color=ps.INK_SECONDARY,
                       fontsize=10)
    axes[0].set_ylabel("recall@k", color=ps.INK_SECONDARY, fontsize=10)
    names = list(conditions)
    x = np.arange(len(names))
    em_v = [conditions[n][0] for n in names]
    f1_v = [conditions[n][1] for n in names]
    axes[1].bar(x - 0.18, em_v, width=0.36, color=ps.SERIES[0], label="EM")
    axes[1].bar(x + 0.18, f1_v, width=0.36, color=ps.SERIES[1], label="F1")
    for xi, v in zip(x - 0.18, em_v):
        axes[1].text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8,
                     color=ps.INK_SECONDARY)
    for xi, v in zip(x + 0.18, f1_v):
        axes[1].text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8,
                     color=ps.INK_SECONDARY)
    axes[1].set_xticks(x, names)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    axes[1].set_title("End-to-end answer quality (same reader)",
                      color=ps.INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "minimal_rag.png"), facecolor=ps.SURFACE,
                bbox_inches="tight")
    print("wrote outputs/minimal_rag.png")


if __name__ == "__main__":
    main()
