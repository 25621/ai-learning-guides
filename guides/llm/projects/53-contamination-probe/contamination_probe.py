"""Contamination probe: find the eval set hiding inside the pretraining data.

The setup is a controlled experiment, because in the real world you never know
the ground truth:

  * "Pretraining corpus" = 1,000 Wikipedia paragraphs (SQuAD v1.1 dev, reused
    from project 43's `rag_lib`).
  * "Benchmark"          = 300 MMLU questions, which do *not* naturally occur
    in that corpus — so the clean false-positive rate is measurable.
  * We then *inject* a known subset of the benchmark back into the corpus at
    controlled perturbation levels (0% = verbatim leak, up to 50% of words
    rewritten = a paraphrased leak), and ask each detector to find them.

Two detectors, both from scratch:

  1. Exact n-gram overlap (the GPT-3 / Llama-style audit): flag an eval item if
     any of its 13-word n-grams appears verbatim anywhere in the corpus.
  2. MinHash + LSH near-duplicate search: 5-word shingles -> 128-permutation
     MinHash signatures -> banded LSH -> estimated Jaccard.

The result is the point of the project: exact matching is a cliff (one rewrite
and it sees nothing), MinHash degrades gracefully. A lab that only greps for
verbatim strings will report "no contamination" on a corpus that is full of
paraphrased leaks.

CPU, ~1 min. No model inference.
"""

import csv
import os
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "43-minimal-rag"))
sys.path.insert(0, os.path.join(HERE, "..", "51-mmlu-re-run"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))
import plot_style as ps  # noqa: E402
from eval_lib import load_mmlu  # noqa: E402
from rag_lib import build_corpus  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

N_EVAL = 360
PERTURB_LEVELS = [0.0, 0.05, 0.08, 0.12, 0.16, 0.25]
N_INJECT_PER_LEVEL = 40      # eval items leaked at each perturbation level
NGRAM_N = 13                 # the classic contamination n-gram width
SHINGLE_K = 5                # word-shingle width for MinHash
N_HASHES = 128
N_BANDS = 32                 # rows = 128/32 = 4  ->  LSH threshold ~ (1/32)^(1/4) = 0.42
JACCARD_THRESHOLD = 0.40

MERSENNE = (1 << 61) - 1


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens, n):
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def item_text(item):
    """Render an MMLU item the way a leaked quiz page would show it."""
    letters = "ABCD"
    body = "".join(f"{letters[i]}. {c} " for i, c in enumerate(item["choices"]))
    return f"{item['question']} {body}Answer: {letters[item['answer']]}"


def perturb(text, frac, rng, vocab):
    """Rewrite `frac` of the words, spread *evenly* through the text.

    Even spacing is the realistic model of light paraphrasing: an editor touches
    every sentence, not one random cluster. It is also what separates the two
    detectors — evenly spaced edits shrink the longest untouched run, so once the
    spacing drops below the exact detector's n-gram width no verbatim n-gram
    survives at all (a cliff), while the shingle *set* still overlaps a lot (a
    graceful decline). Random-cluster edits would leave long clean runs and hide
    the difference."""
    w = text.split()
    if frac <= 0 or len(w) == 0:
        return text
    k = max(1, int(round(frac * len(w))))
    # evenly spaced positions across the item
    idx = np.linspace(0, len(w) - 1, k, dtype=int) if k > 1 else [len(w) // 2]
    for i in idx:
        w[i] = vocab[rng.integers(0, len(vocab))]
    return " ".join(w)


# --------------------------------------------------------------------------- #
# Detector 1 — exact n-gram overlap
# --------------------------------------------------------------------------- #
class ExactNgramIndex:
    def __init__(self, docs, n=NGRAM_N):
        self.n = n
        self.grams = set()
        for d in docs:
            self.grams |= ngrams(words(d), n)

    def flags(self, text):
        return len(ngrams(words(text), self.n) & self.grams) > 0


# --------------------------------------------------------------------------- #
# Detector 2 — MinHash + LSH, from scratch
# --------------------------------------------------------------------------- #
class MinHashLSH:
    """128-permutation MinHash over word shingles, indexed by banded LSH.

    MinHash: for a random permutation h of the shingle universe, the probability
    that two sets share a minimum is exactly their Jaccard similarity. Average
    over 128 permutations and you get an unbiased Jaccard estimate for the price
    of 128 integers per document.

    LSH: split the 128-slot signature into `n_bands` bands of `rows` slots. Two
    documents become candidates if *any* band matches exactly, which happens with
    probability 1-(1-J^rows)^n_bands — an S-curve that is steep around
    (1/n_bands)^(1/rows), so we only ever compare plausible pairs.
    """

    def __init__(self, n_hashes=N_HASHES, n_bands=N_BANDS, k=SHINGLE_K, seed=0):
        assert n_hashes % n_bands == 0
        self.n_hashes, self.n_bands = n_hashes, n_bands
        self.rows = n_hashes // n_bands
        self.k = k
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MERSENNE, size=n_hashes, dtype=np.int64)
        self.b = rng.integers(0, MERSENNE, size=n_hashes, dtype=np.int64)
        self.buckets = [dict() for _ in range(n_bands)]
        self.sigs = {}

    def shingles(self, text):
        w = words(text)
        if len(w) < self.k:
            return {hash(tuple(w)) & 0xFFFFFFFF} if w else set()
        return {
            hash(tuple(w[i : i + self.k])) & 0xFFFFFFFF
            for i in range(len(w) - self.k + 1)
        }

    def signature(self, text):
        sh = self.shingles(text)
        if not sh:
            return np.full(self.n_hashes, MERSENNE, dtype=np.int64)
        x = np.fromiter(sh, dtype=np.int64, count=len(sh))[:, None]  # (S,1)
        # (a*x + b) mod prime, one column per permutation -> min over shingles
        h = (self.a[None, :] * x + self.b[None, :]) % MERSENNE
        return h.min(axis=0)

    def add(self, doc_id, text):
        sig = self.signature(text)
        self.sigs[doc_id] = sig
        for bi in range(self.n_bands):
            band = tuple(sig[bi * self.rows : (bi + 1) * self.rows].tolist())
            self.buckets[bi].setdefault(band, []).append(doc_id)

    def query(self, text):
        """Return (best_estimated_jaccard, best_doc_id) over LSH candidates."""
        sig = self.signature(text)
        cands = set()
        for bi in range(self.n_bands):
            band = tuple(sig[bi * self.rows : (bi + 1) * self.rows].tolist())
            cands.update(self.buckets[bi].get(band, ()))
        best, best_id = 0.0, None
        for c in cands:
            j = float(np.mean(self.sigs[c] == sig))  # MinHash Jaccard estimate
            if j > best:
                best, best_id = j, c
        return best, best_id


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    rng = np.random.default_rng(0)

    paragraphs, _ = build_corpus(n_paragraphs=1000, n_questions=10, seed=0)
    corpus = [p["text"] for p in paragraphs]
    vocab = sorted({w for d in corpus[:200] for w in words(d)})
    print(f"corpus: {len(corpus)} paragraphs, vocab sample {len(vocab)} words")

    # Keep only items long enough to hold several 13-grams, so every item is
    # detectable in principle and the geometry (edits per n-gram) is uniform —
    # short items otherwise add non-monotone noise to the recall curve.
    all_items = load_mmlu(per_subject=12)
    eval_items = [it for it in all_items if len(words(item_text(it))) >= 35][:N_EVAL]
    eval_texts = [item_text(it) for it in eval_items]
    print(f"benchmark: {len(eval_texts)} MMLU items (>=35 words each)")

    # ---- inject contamination at known perturbation levels ---------------- #
    # Each leaked item enters the corpus as its own document — a quiz-site page
    # scraped into the crawl. (Contaminants buried inside much larger documents
    # dilute the MinHash signal; that harder case is left as a "things to try".)
    leaked = {}   # eval index -> perturbation level
    pool = rng.permutation(len(eval_texts))
    cursor = 0
    contaminated = list(corpus)
    for level in PERTURB_LEVELS:
        for _ in range(N_INJECT_PER_LEVEL):
            ei = int(pool[cursor]); cursor += 1
            leaked[ei] = level
            contaminated.append(perturb(eval_texts[ei], level, rng, vocab))
    clean_idx = [i for i in range(len(eval_texts)) if i not in leaked]
    print(f"injected {len(leaked)} leaks across {len(PERTURB_LEVELS)} levels; "
          f"{len(clean_idx)} eval items stay clean\n")

    # ---- build both indexes ---------------------------------------------- #
    t = time.time()
    exact = ExactNgramIndex(contaminated)
    t_exact_build = time.time() - t

    t = time.time()
    lsh = MinHashLSH()
    for i, d in enumerate(contaminated):
        lsh.add(i, d)
    t_lsh_build = time.time() - t
    print(f"index build: exact {NGRAM_N}-gram {t_exact_build:.1f}s "
          f"({len(exact.grams):,} grams) | MinHash-LSH {t_lsh_build:.1f}s")

    # ---- run both detectors over every eval item -------------------------- #
    t = time.time()
    exact_hit = [exact.flags(t_) for t_ in eval_texts]
    t_exact_q = time.time() - t

    t = time.time()
    jac = [lsh.query(t_)[0] for t_ in eval_texts]
    t_lsh_q = time.time() - t
    minhash_hit = [j >= JACCARD_THRESHOLD for j in jac]
    print(f"query {len(eval_texts)} items: exact {t_exact_q:.1f}s | "
          f"MinHash {t_lsh_q:.1f}s\n")

    # ---- recall per perturbation level, FP rate on clean items ------------ #
    rows = []
    for level in PERTURB_LEVELS:
        ids = [i for i, lv in leaked.items() if lv == level]
        er = float(np.mean([exact_hit[i] for i in ids]))
        mr = float(np.mean([minhash_hit[i] for i in ids]))
        mj = float(np.mean([jac[i] for i in ids]))
        rows.append((level, er, mr, mj))
        print(f"  perturb {level:4.0%}  exact recall {er:.2f}   "
              f"MinHash recall {mr:.2f}   (mean Ĵ={mj:.2f})")

    fp_exact = float(np.mean([exact_hit[i] for i in clean_idx]))
    fp_minhash = float(np.mean([minhash_hit[i] for i in clean_idx]))
    mean_j_clean = float(np.mean([jac[i] for i in clean_idx]))
    print(f"\nfalse positives on clean items: exact {fp_exact:.3f}  "
          f"MinHash {fp_minhash:.3f}  (mean Ĵ on clean = {mean_j_clean:.3f})")

    with open(os.path.join(OUT, "contamination.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["perturb_frac", "exact_recall", "minhash_recall", "mean_jaccard"])
        for r in rows:
            w.writerow([f"{r[0]:.2f}", f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.3f}"])
        w.writerow([])
        w.writerow(["clean_fp_exact", f"{fp_exact:.3f}"])
        w.writerow(["clean_fp_minhash", f"{fp_minhash:.3f}"])

    # ---- figure 1: recall vs perturbation --------------------------------- #
    fig, ax = ps.new_axes(7.6, 4.4)
    x = [r[0] * 100 for r in rows]
    ax.plot(x, [r[1] for r in rows], "-o", color=ps.SERIES[2], lw=2, ms=6,
            zorder=3, label=f"exact {NGRAM_N}-gram recall")
    ax.plot(x, [r[2] for r in rows], "-o", color=ps.SERIES[0], lw=2, ms=6,
            zorder=3, label="MinHash-LSH recall")
    ax.plot(x, [r[3] for r in rows], "--", color=ps.SERIES[1], lw=1.6, zorder=2,
            label="MinHash graded similarity (mean Ĵ)")
    ax.axhline(JACCARD_THRESHOLD, color=ps.BASELINE, lw=1, ls=":", zorder=1)
    ax.text(max(x), JACCARD_THRESHOLD + 0.015, f"flag threshold Ĵ={JACCARD_THRESHOLD}",
            ha="right", va="bottom", color=ps.INK_MUTED, fontsize=9)
    # false-positive markers at x=0 (the clean-item flag rates)
    ax.scatter([0], [fp_exact], marker="x", s=55, color=ps.SERIES[2], zorder=5)
    ax.annotate(f"exact false-positive rate {fp_exact:.0%}\n(shared boilerplate)",
                xy=(0, fp_exact), xytext=(4, 0.20), fontsize=8.5,
                color=ps.SERIES[2],
                arrowprops=dict(arrowstyle="->", color=ps.SERIES[2], lw=0.8))
    # shade the regime where both string detectors go blind
    ax.axvspan(18, max(x), color=ps.INK_MUTED, alpha=0.06, zorder=0)
    ax.text(max(x) - 0.5, 0.62, "heavy paraphrase:\nboth string methods blind\n"
            "(needs semantic detection)", ha="right", va="top", fontsize=8.5,
            color=ps.INK_MUTED)
    ax.set_xlabel("% of the leaked item's words rewritten (spread evenly)")
    ax.set_ylabel("detection recall")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Exact match is a binary cliff with false alarms; MinHash is a "
                 "graded score", color=ps.INK, fontsize=11.5, loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "detection_vs_perturbation.png"))

    # ---- figure 2: the score distribution that makes the threshold work ---- #
    fig, ax = ps.new_axes(7.2, 4.0)
    clean_j = [jac[i] for i in clean_idx]
    leak_j = [jac[i] for i in leaked]
    bins = np.linspace(0, 1, 41)
    ax.hist(clean_j, bins=bins, color=ps.SERIES[1], alpha=0.75, zorder=3,
            label=f"clean items (n={len(clean_j)})")
    ax.hist(leak_j, bins=bins, color=ps.SERIES[2], alpha=0.75, zorder=3,
            label=f"leaked items (n={len(leak_j)})")
    ax.axvline(JACCARD_THRESHOLD, color=ps.INK, lw=1.4, ls="--", zorder=4)
    ax.text(JACCARD_THRESHOLD + 0.01, ax.get_ylim()[1] * 0.9, "flag threshold",
            color=ps.INK, fontsize=9)
    ax.set_xlabel("estimated Jaccard against the nearest corpus document")
    ax.set_ylabel("eval items")
    ax.set_yscale("log")
    ax.set_title("The separation a contamination audit actually relies on",
                 color=ps.INK, fontsize=12, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "score_separation.png"))

    print(f"\ndone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
