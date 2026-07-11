"""Project 45 — Reranker effect: bi-encoder top-20, cross-encoder re-scores.

Stage 1 (fast, rough): MiniLM bi-encoder — the query and every paragraph were
embedded *independently*, so matching is a single dot product per document.
Stage 2 (slow, sharp): a MS-MARCO cross-encoder reads query+paragraph
*together* and outputs a relevance score; we re-sort stage 1's top-20 by it.

Measured: nDCG@10 / recall@k before vs after, end-to-end EM/F1 with the top-3
contexts, and the per-query latency of each stage — the whole point of the
two-stage design is that the accurate scorer is too slow to run on everything.
"""

import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "43-minimal-rag"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))

import plot_style as ps  # noqa: E402
from rag_lib import (RERANKER_ID, Embedder, Reader, build_corpus,  # noqa: E402
                     dense_search, em_score, f1_score, ndcg_at_k, recall_at_k)

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
N_FIRST_STAGE = 20
K_CTX = 3


class CrossEncoder:
    def __init__(self):
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        self.tok = AutoTokenizer.from_pretrained(RERANKER_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_ID).eval()

    @torch.no_grad()
    def score(self, query, passages, batch_size=32):
        out = []
        for i in range(0, len(passages), batch_size):
            enc = self.tok([query] * len(passages[i:i + batch_size]),
                           passages[i:i + batch_size], padding=True,
                           truncation=True, max_length=256,
                           return_tensors="pt")
            out.append(self.model(**enc).logits.squeeze(-1))
        return torch.cat(out).numpy()


def main():
    paragraphs, questions = build_corpus(n_paragraphs=1000, n_questions=300)
    texts = [p["text"] for p in paragraphs]

    embedder = Embedder()
    doc_vecs = embedder.encode(texts, verbose=True)
    t0 = time.time()
    q_vecs = embedder.encode([q["q"] for q in questions])
    top, _ = dense_search(q_vecs, doc_vecs, k=N_FIRST_STAGE)
    dense_ms = (time.time() - t0) / len(questions) * 1000

    print("== reranking ==", flush=True)
    ce = CrossEncoder()
    reranked = []
    t0 = time.time()
    for i, q in enumerate(questions):
        cand = list(top[i])
        scores = ce.score(q["q"], [texts[j] for j in cand])
        reranked.append([cand[j] for j in np.argsort(-scores)])
        if i % 100 == 0:
            print(f"  reranked {i}/{len(questions)}", flush=True)
    ce_ms = (time.time() - t0) / len(questions) * 1000

    stages = {"bi-encoder": [list(t) for t in top], "+ reranker": reranked}
    ks = [1, 3, 5]
    metrics = {}
    for name, ranks in stages.items():
        rec = {k: np.mean([recall_at_k(ranks[i], q["gold"], k)
                           for i, q in enumerate(questions)]) for k in ks}
        ndcg = np.mean([ndcg_at_k(ranks[i], q["gold"], 10)
                        for i, q in enumerate(questions)])
        metrics[name] = (rec, ndcg)
        print(f"{name:11s} nDCG@10 {ndcg:.3f}  " +
              "  ".join(f"R@{k} {rec[k]:.3f}" for k in ks))

    print("== end-to-end ==", flush=True)
    reader = Reader()
    e2e = {}
    for name, ranks in stages.items():
        em, f1 = [], []
        for i, q in enumerate(questions):
            pred, _ = reader.answer(
                q["q"], [texts[j] for j in ranks[i][:K_CTX]])
            em.append(em_score(pred, q["answers"]))
            f1.append(f1_score(pred, q["answers"]))
            if i % 100 == 0:
                print(f"  {name}: {i}/{len(questions)}", flush=True)
        e2e[name] = (np.mean(em), np.mean(f1))
        print(f"  {name:11s} EM {np.mean(em):.3f}  F1 {np.mean(f1):.3f}")

    print(f"latency/query: bi-encoder {dense_ms:.1f} ms, "
          f"cross-encoder(20 docs) {ce_ms:.0f} ms "
          f"({ce_ms / dense_ms:.0f}x)")

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "ndcg@10", "recall@1", "recall@3", "recall@5",
                    "EM", "F1", "ms_per_query"])
        for name in stages:
            rec, ndcg = metrics[name]
            ms = dense_ms if name == "bi-encoder" else dense_ms + ce_ms
            w.writerow([name, f"{ndcg:.3f}"] +
                       [f"{rec[k]:.3f}" for k in ks] +
                       [f"{e2e[name][0]:.3f}", f"{e2e[name][1]:.3f}",
                        f"{ms:.1f}"])

    # figure: grouped bars for ranking + answer metrics, both stages
    names = list(stages)
    cats = ["nDCG@10", "recall@1", "recall@3", "EM", "F1"]
    vals = {
        names[0]: [metrics[names[0]][1], metrics[names[0]][0][1],
                   metrics[names[0]][0][3], e2e[names[0]][0],
                   e2e[names[0]][1]],
        names[1]: [metrics[names[1]][1], metrics[names[1]][0][1],
                   metrics[names[1]][0][3], e2e[names[1]][0],
                   e2e[names[1]][1]],
    }
    fig, ax = ps.new_axes(8.6, 4.4)
    x = np.arange(len(cats))
    for s, (name, color) in enumerate(zip(names, ps.SERIES)):
        off = (s - 0.5) * 0.38
        ax.bar(x + off, vals[name], width=0.38, color=color, label=name)
        for xi, v in zip(x + off, vals[name]):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8,
                    color=ps.INK_SECONDARY)
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax,
              "Cross-encoder reranking of the bi-encoder's top-20",
              "metric", "score", os.path.join(OUT, "reranker_effect.png"))


if __name__ == "__main__":
    main()
