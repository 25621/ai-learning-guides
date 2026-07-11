"""Project 46 — Hybrid retrieval: dense + BM25 fused with reciprocal rank fusion.

Three retrievers over the same 1,000 Wikipedia paragraphs:

  dense  — MiniLM embeddings, cosine similarity (matches *meaning*)
  BM25   — from-scratch Okapi BM25 (matches *exact words*)
  hybrid — RRF over the two ranked lists: score(d) = sum 1/(60 + rank)

Besides the headline metrics, the interesting object is the disagreement set:
queries only one of the two base retrievers gets right. RRF wins iff it
rescues most of both one-sided sets without breaking the agreed ones.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "43-minimal-rag"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))

import plot_style as ps  # noqa: E402
from rag_lib import (BM25, Embedder, Reader, build_corpus, dense_search,  # noqa: E402
                     em_score, f1_score, ndcg_at_k, recall_at_k, rrf_fuse)

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
K_CTX = 3


def main():
    paragraphs, questions = build_corpus(n_paragraphs=1000, n_questions=300)
    texts = [p["text"] for p in paragraphs]

    embedder = Embedder()
    doc_vecs = embedder.encode(texts, verbose=True)
    q_vecs = embedder.encode([q["q"] for q in questions])
    dense_top, _ = dense_search(q_vecs, doc_vecs, k=20)

    print("== BM25 ==", flush=True)
    bm25 = BM25(texts)
    bm25_top = [bm25.search(q["q"], 20)[0] for q in questions]

    hybrid_top = [rrf_fuse([list(dense_top[i]), list(bm25_top[i])],
                           k=60, top_k=20)
                  for i in range(len(questions))]

    systems = {"dense": [list(t) for t in dense_top],
               "BM25": [list(t) for t in bm25_top],
               "hybrid (RRF)": hybrid_top}

    ks = [1, 5]
    reader = Reader()
    rows, table = [], {}
    for name, ranks in systems.items():
        rec = {k: np.mean([recall_at_k(ranks[i], q["gold"], k)
                           for i, q in enumerate(questions)]) for k in ks}
        ndcg = np.mean([ndcg_at_k(ranks[i], q["gold"], 10)
                        for i, q in enumerate(questions)])
        em, f1 = [], []
        for i, q in enumerate(questions):
            pred, _ = reader.answer(
                q["q"], [texts[j] for j in ranks[i][:K_CTX]])
            em.append(em_score(pred, q["answers"]))
            f1.append(f1_score(pred, q["answers"]))
        table[name] = (rec, ndcg, np.mean(em), np.mean(f1))
        print(f"{name:12s} R@1 {rec[1]:.3f}  R@5 {rec[5]:.3f}  "
              f"nDCG@10 {ndcg:.3f}  EM {np.mean(em):.3f}  "
              f"F1 {np.mean(f1):.3f}", flush=True)
        rows.append([name, f"{rec[1]:.3f}", f"{rec[5]:.3f}", f"{ndcg:.3f}",
                     f"{np.mean(em):.3f}", f"{np.mean(f1):.3f}"])

    # disagreement analysis at k=5
    def hit(ranks, i):
        return recall_at_k(ranks[i], questions[i]["gold"], 5) > 0
    only_dense = [i for i in range(len(questions))
                  if hit(systems["dense"], i) and not hit(systems["BM25"], i)]
    only_bm25 = [i for i in range(len(questions))
                 if hit(systems["BM25"], i) and not hit(systems["dense"], i)]
    both = sum(1 for i in range(len(questions))
               if hit(systems["dense"], i) and hit(systems["BM25"], i))
    rescued_d = sum(1 for i in only_bm25 if hit(systems["hybrid (RRF)"], i))
    rescued_b = sum(1 for i in only_dense if hit(systems["hybrid (RRF)"], i))
    print(f"@5: both {both}, only-dense {len(only_dense)}, "
          f"only-BM25 {len(only_bm25)}")
    print(f"hybrid keeps {rescued_b}/{len(only_dense)} dense-only wins and "
          f"{rescued_d}/{len(only_bm25)} BM25-only wins")
    print("sample dense-only wins (paraphrase-ish):")
    for i in only_dense[:3]:
        print(f"  {questions[i]['q']}")
    print("sample BM25-only wins (exact-term-ish):")
    for i in only_bm25[:3]:
        print(f"  {questions[i]['q']}")

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "recall@1", "recall@5", "ndcg@10", "EM", "F1"])
        w.writerows(rows)
        w.writerow(["only_dense@5", len(only_dense), "rescued_by_hybrid",
                    rescued_b, "", ""])
        w.writerow(["only_bm25@5", len(only_bm25), "rescued_by_hybrid",
                    rescued_d, "", ""])

    # figure
    names = list(systems)
    cats = ["recall@1", "recall@5", "nDCG@10", "F1"]
    fig, ax = ps.new_axes(8.6, 4.4)
    x = np.arange(len(cats))
    for s, name in enumerate(names):
        rec, ndcg, em, f1 = table[name]
        vals = [rec[1], rec[5], ndcg, f1]
        off = (s - 1) * 0.27
        ax.bar(x + off, vals, width=0.27, color=ps.SERIES[s], label=name)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5,
                    color=ps.INK_SECONDARY)
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY,
              ncols=3)
    ps.finish(fig, ax, "Dense vs BM25 vs reciprocal-rank fusion",
              "metric", "score", os.path.join(OUT, "hybrid_retrieval.png"))


if __name__ == "__main__":
    main()
