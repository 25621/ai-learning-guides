"""Project 44 — Chunking ablation: 200 vs 800 vs 1,600-token chunks, ± overlap.

The 48 SQuAD articles are glued back into full documents (5k-9k tokens each),
then re-chunked at several sizes. Every config is evaluated at the SAME
retrieved-context budget (~1,600 tokens handed to the reader: 8x200, 2x800,
or 1x1600), so the question is purely *how should a fixed context budget be
sliced*, not *who gets more text*.

Ground truth per question: a chunk is "relevant" iff it contains the gold
answer span (exact char offsets from SQuAD, mapped through the chunker).

The trap this ablation exposes: the embedder (MiniLM) reads at most 512
wordpieces. A 1,600-token chunk is indexed by its first third only — the rest
is retrievable by luck alone.
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "43-minimal-rag"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))

import plot_style as ps  # noqa: E402
from rag_lib import Embedder, Reader, f1_score, load_squad, ndcg_at_k  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

BUDGET = 1600            # reader context budget in tokens, all configs equal
N_QUESTIONS = 150
CONFIGS = [(200, 0), (200, 50), (800, 0), (800, 200), (1600, 0)]


def build_documents():
    """Concatenate each article's paragraphs; keep absolute answer offsets."""
    rng = np.random.default_rng(0)
    docs, questions = [], []
    for art in load_squad():
        doc_id = len(docs)
        parts, offset = [], 0
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                a = qa["answers"][0]
                questions.append({
                    "q": qa["question"].strip(),
                    "answers": [x["text"] for x in qa["answers"]],
                    "doc": doc_id,
                    "span": (offset + a["answer_start"],
                             offset + a["answer_start"] + len(a["text"])),
                })
            parts.append(para["context"])
            offset += len(para["context"]) + 2  # the "\n\n" joiner
        docs.append("\n\n".join(parts))
    pick = rng.permutation(len(questions))[:N_QUESTIONS]
    return docs, [questions[i] for i in pick]


def chunk_documents(docs, tokenizer, size, overlap):
    """Token-window chunking; returns chunk texts + (doc, char-range) spans."""
    texts, spans = [], []
    step = size - overlap
    for doc_id, doc in enumerate(docs):
        enc = tokenizer(doc, add_special_tokens=False,
                        return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        for start in range(0, len(offs), step):
            window = offs[start:start + size]
            lo, hi = window[0][0], window[-1][1]
            texts.append(doc[lo:hi])
            spans.append((doc_id, lo, hi))
            if start + size >= len(offs):
                break
    return texts, spans


def main():
    docs, questions = build_documents()
    print(f"{len(docs)} documents, {N_QUESTIONS} questions")
    embedder = Embedder(max_len=512)  # give big chunks the encoder's full cap
    reader = Reader()
    q_vecs = embedder.encode([q["q"] for q in questions])

    rows = []
    for size, overlap in CONFIGS:
        name = f"{size}tok" + (f"+ov{overlap}" if overlap else "")
        t0 = time.time()
        texts, spans = chunk_documents(docs, embedder.tok, size, overlap)
        vecs = embedder.encode(texts)
        embed_s = time.time() - t0
        n_ctx = max(1, BUDGET // size)

        sims = q_vecs @ vecs.T
        order = np.argsort(-sims, axis=1)
        recall, ndcg, f1s = [], [], []
        for i, q in enumerate(questions):
            relevant = {j for j, (d, lo, hi) in enumerate(spans)
                        if d == q["doc"] and lo <= q["span"][0]
                        and q["span"][1] <= hi}
            top_budget = order[i][:n_ctx]
            recall.append(float(any(j in relevant for j in top_budget)))
            gold_ranked = [j for j in order[i][:10] if j in relevant]
            ndcg.append(ndcg_at_k(list(order[i][:10]),
                                  gold_ranked[0] if gold_ranked else -1, 10))
            pred, _ = reader.answer(q["q"], [texts[j] for j in top_budget])
            f1s.append(f1_score(pred, q["answers"]))
            if i % 100 == 0:
                print(f"  [{name}] q {i}/{len(questions)}", flush=True)
        rows.append([name, size, overlap, len(texts), n_ctx,
                     f"{np.mean(recall):.3f}", f"{np.mean(ndcg):.3f}",
                     f"{np.mean(f1s):.3f}", f"{embed_s:.0f}"])
        print(f"{name:12s} chunks={len(texts):5d} ctx={n_ctx} "
              f"recall@budget={np.mean(recall):.3f} nDCG@10={np.mean(ndcg):.3f} "
              f"F1={np.mean(f1s):.3f} embed={embed_s:.0f}s", flush=True)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "size", "overlap", "n_chunks", "ctx_chunks",
                    "recall@budget", "ndcg@10", "f1", "embed_s"])
        w.writerows(rows)

    # figure: recall@budget and end-to-end F1 per config
    labels = [r[0] for r in rows]
    rec = [float(r[5]) for r in rows]
    f1v = [float(r[7]) for r in rows]
    fig, ax = ps.new_axes(8.6, 4.4)
    x = np.arange(len(labels))
    ax.bar(x - 0.19, rec, width=0.38, color=ps.SERIES[0],
           label="answer span retrieved (within 1,600-token budget)")
    ax.bar(x + 0.19, f1v, width=0.38, color=ps.SERIES[1],
           label="end-to-end answer F1")
    for xi, v in zip(x - 0.19, rec):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8,
                color=ps.INK_SECONDARY)
    for xi, v in zip(x + 0.19, f1v):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8,
                color=ps.INK_SECONDARY)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Chunk size at a fixed 1,600-token context budget",
              "chunking config", "score",
              os.path.join(OUT, "chunking_ablation.png"))


if __name__ == "__main__":
    main()
