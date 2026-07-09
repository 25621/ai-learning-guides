"""Dedup ablation: two identical models, the only difference is whether the
duplicates were removed first.

Real web crawls are dominated by *skewed* duplication — a small set of documents
(boilerplate, popular pages, spam templates) appears hundreds of times, while the
long tail appears once. Under a fixed compute budget that skew is corrosive: the
model spends most of its gradient steps memorizing the few over-represented
documents and under-trains on everything else, so it generalizes worse. We build
exactly that pathology — take a clean corpus, pick a small "hot" subset, and
duplicate each hot document many times — then remove the duplicates with a
from-scratch MinHash and train two identical ~1.9M models for the same number of
steps. We measure both sides: how well each generalizes to held-out clean text,
and how hard each has memorized the hot documents.

    python dedup_ablation.py --run          # build corpora, train both, evaluate
    python dedup_ablation.py --plot         # figures from the saved results

Reuses the GPT skeleton from project 08.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT, cosine_lr   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
DOC_LEN = 1500
STEPS = 600
BATCH = 24
BLOCK = 128
HOT_FRAC = 0.05              # fraction of documents that are "hot" (over-represented)
DUP_TIMES = 30              # how many extra copies of each hot document the crawl has


# --------------------------------------------------------------- MinHash dedup
def shingles(doc, k=5):
    words = doc.split()
    if len(words) < k:
        return {doc}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def minhash_signature(shs, seeds, mask=(1 << 32) - 1):
    """One min-hash per seed: min over shingles of a seeded 32-bit hash."""
    sig = np.full(len(seeds), mask, dtype=np.uint64)
    for sh in shs:
        h = hash(sh) & mask
        # cheap independent hash family: (a*h + b) mod 2^32
        vals = (seeds[:, 0] * h + seeds[:, 1]) & mask
        sig = np.minimum(sig, vals)
    return sig


def dedup(docs, n_hashes=64, bands=16, thresh=0.7, seed=0):
    """LSH-banded MinHash near-dedup. Returns indices of kept documents."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, (1 << 32) - 1, size=(n_hashes, 2)).astype(np.uint64)
    sigs = [minhash_signature(shingles(d), seeds) for d in docs]
    rows = n_hashes // bands
    buckets = {}
    for i, sig in enumerate(sigs):
        for b in range(bands):
            band = tuple(int(x) for x in sig[b * rows:(b + 1) * rows])
            buckets.setdefault((b, band), []).append(i)
    # union-find over candidate pairs that share a band, confirmed by signature sim
    parent = list(range(len(docs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for members in buckets.values():
        for j in range(1, len(members)):
            a, b = members[0], members[j]
            sim = np.mean(sigs[a] == sigs[b])
            if sim >= thresh:
                union(a, b)
    keep = {}
    for i in range(len(docs)):
        r = find(i)
        if r not in keep:
            keep[r] = i
    return sorted(keep.values())


# --------------------------------------------------------------- data plumbing
def make_corpora(seed=0):
    text = Path(HERE / "data/corpus.txt").read_text(encoding="utf-8")
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    docs = [text[i:i + DOC_LEN] for i in range(0, len(text) - DOC_LEN, DOC_LEN)]
    rng = np.random.default_rng(seed)
    rng.shuffle(docs)
    n_val = int(0.15 * len(docs))
    clean_val = docs[:n_val]                       # held out from BOTH training sets
    train_docs = docs[n_val:]

    # simulate a skewed crawl: a small "hot" subset is duplicated many times,
    # with a light per-copy perturbation so copies are *near*-duplicates.
    n_hot = max(1, int(HOT_FRAC * len(train_docs)))
    hot = train_docs[:n_hot]
    raw = list(train_docs)
    for d in hot:
        for _ in range(DUP_TIMES):
            chars = list(d)
            for _ in range(rng.integers(1, 6)):        # flip a handful of chars
                p = rng.integers(0, len(chars))
                chars[p] = rng.choice(list(stoi))
            raw.append("".join(chars))
    rng.shuffle(raw)
    kept = dedup(raw)
    deduped = [raw[i] for i in kept]
    return stoi, clean_val, train_docs, hot, raw, deduped


def encode(docs, stoi):
    joined = "\n".join(docs)
    return torch.tensor([stoi[c] for c in joined if c in stoi], dtype=torch.long)


def batch(t, bs):
    ix = torch.randint(len(t) - BLOCK - 1, (bs,))
    x = torch.stack([t[i:i + BLOCK] for i in ix])
    y = torch.stack([t[i + 1:i + 1 + BLOCK] for i in ix])
    return x, y


@torch.no_grad()
def eval_loss(model, t, iters=40):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = batch(t, BATCH)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


def train_on(tensor, vocab, tag):
    torch.manual_seed(0); torch.set_num_threads(12)
    cfg = Config(vocab_size=vocab, n_layer=4, n_head=4, n_embd=128, block_size=BLOCK)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    t0 = time.time()
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, STEPS, 3e-3, warmup=80)
        x, y = batch(tensor, BATCH)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 150 == 0:
            print(f"  [{tag}] step {step}/{STEPS} loss {loss.item():.3f} "
                  f"({step/(time.time()-t0+1e-9):.2f} it/s)", flush=True)
    return model


def run():
    CKPT.mkdir(parents=True, exist_ok=True)
    stoi, clean_val, train_docs, hot, raw, deduped = make_corpora()
    vocab = len(stoi)
    val_t = encode(clean_val, stoi)                    # generalization probe (held out)
    hot_t = encode(hot, stoi)                          # memorization probe (over-represented docs)
    raw_t, dedup_t = encode(raw, stoi), encode(deduped, stoi)
    hot_share = 1 - len(deduped) / len(raw)
    print(f"docs: {len(train_docs)} unique | {len(hot)} hot x{DUP_TIMES} copies "
          f"| raw crawl {len(raw)} | after MinHash {len(deduped)} "
          f"({100*hot_share:.0f}% of the crawl was duplicate)")
    print(f"tokens: raw {len(raw_t)/1e6:.2f}M | deduped {len(dedup_t)/1e6:.2f}M "
          f"| clean val {len(val_t)/1e3:.0f}k | hot {len(hot_t)/1e3:.0f}k")

    results = {}
    for tag, tensor in [("raw", raw_t), ("deduped", dedup_t)]:
        model = train_on(tensor, vocab, tag)
        clean = eval_loss(model, val_t)
        hotl = eval_loss(model, hot_t)                 # loss on the over-represented docs
        results[tag] = (clean, hotl)
        print(f"[{tag}] clean-val loss {clean:.3f} | hot-doc loss {hotl:.3f}")

    np.savez(CKPT / "results.npz",
             raw_clean=results["raw"][0], raw_hot=results["raw"][1],
             dedup_clean=results["deduped"][0], dedup_hot=results["deduped"][1],
             n_raw=len(raw), n_dedup=len(deduped),
             tok_raw=len(raw_t), tok_dedup=len(dedup_t), hot_share=hot_share)
    print("wrote checkpoints/results.npz")


def plot():
    import plot_style as ps
    OUT.mkdir(parents=True, exist_ok=True)
    r = np.load(CKPT / "results.npz")

    fig, ax = ps.new_axes(7.2, 4.4)
    groups = ["clean-val loss\n(generalization — lower is better)",
              "hot-doc loss\n(memorization — lower = more memorized)"]
    raw_vals = [float(r["raw_clean"]), float(r["raw_hot"])]
    ded_vals = [float(r["dedup_clean"]), float(r["dedup_hot"])]
    x = np.arange(len(groups)); w = 0.36
    ax.bar(x - w / 2, raw_vals, w, color=ps.SERIES[2], label="raw (skewed duplicates)")
    ax.bar(x + w / 2, ded_vals, w, color=ps.SERIES[1], label="deduped (MinHash)")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
    for xi, v in zip(x - w / 2, raw_vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8, color=ps.INK_SECONDARY)
    for xi, v in zip(x + w / 2, ded_vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8, color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Duplicates buy memorization at the cost of generalization",
              "", "cross-entropy loss (nats/char)", OUT / "dedup_effect.png")

    with open(OUT / "results.csv", "w") as f:
        f.write("corpus,documents,tokens_millions,clean_val_loss,hot_doc_loss\n")
        f.write(f"raw,{int(r['n_raw'])},{float(r['tok_raw'])/1e6:.2f},"
                f"{float(r['raw_clean']):.3f},{float(r['raw_hot']):.3f}\n")
        f.write(f"deduped,{int(r['n_dedup'])},{float(r['tok_dedup'])/1e6:.2f},"
                f"{float(r['dedup_clean']):.3f},{float(r['dedup_hot']):.3f}\n")
    print(f"wrote {OUT/'dedup_effect.png'} + results.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.run:
        run()
    if args.plot:
        plot()


if __name__ == "__main__":
    main()
