"""Quality-filter ablation: does throwing away the junk actually help?

We build a "crawl" that is half good text (tiny-shakespeare) and half synthetic
junk (character noise, keyboard mash, single-word spam, tag soup). A from-scratch
logistic-regression quality classifier — trained on a handful of cheap surface
features, FineWeb-Edu style — scores every document and keeps only the ones it
judges educational. Two identical ~1.9M models then train for the same number of
steps: one on the raw mix, one on the filtered corpus. Both are judged on the
*same* held-out clean text.

    python quality_filter.py --run       # build corpus, train classifier + both LMs
    python quality_filter.py --plot      # figures from saved results

Reuses the GPT skeleton from project 08.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT, cosine_lr   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
DOC_LEN = 1200
STEPS = 600
BATCH = 24
BLOCK = 128
JUNK_CHARS = "@#$%^&*_=+{}[]|\\/<>~`"      # symbols rare in clean prose


# --------------------------------------------------------------- junk factory
def make_junk(kind, n, rng, vocab_chars):
    if kind == "noise":                                    # uniform character soup
        chars = list(vocab_chars)
        idx = rng.integers(0, len(chars), n)
        return "".join(chars[i] for i in idx)
    if kind == "mash":                                     # repeated keyboard rows
        rows = ["asdfghjkl ", "qwertyuiop ", "zxcvbnm "]
        return "".join(rows[i] for i in rng.integers(0, 3, n // 8))[:n]
    if kind == "repeat":                                   # one word, over and over
        letters = list("abcdefghijklmnopqrstuvwxyz")
        w = "".join(letters[i] for i in rng.integers(0, 26, rng.integers(3, 8)))
        return ((w + " ") * (n // (len(w) + 1)))[:n]
    if kind == "tagsoup":                                  # SEO / markup spam
        toks = ["<div>", "</div>", "BUY", "NOW", "CLICK", "$$$", "<a>", "###", "SALE"]
        return " ".join(toks[i] for i in rng.integers(0, len(toks), n // 5))[:n]
    return ""


# --------------------------------------------------- surface quality features
def features(doc):
    n = max(1, len(doc))
    alpha = sum(c.isalpha() or c.isspace() for c in doc) / n     # fraction letters/space
    symbol = sum(c in JUNK_CHARS for c in doc) / n               # junk-symbol density
    words = doc.split()
    mean_wlen = np.mean([len(w) for w in words]) if words else 0
    uniq = len(set(words)) / max(1, len(words))                  # lexical diversity
    top_char = max((doc.count(c) for c in set(doc)), default=0) / n
    upper = sum(c.isupper() for c in doc) / n
    return np.array([alpha, symbol, mean_wlen / 10, uniq, top_char, upper])


# --------------------------------------------------- logistic-regression head
def train_classifier(X, y, epochs=400, lr=0.5):
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-8)
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(epochs):
        z = Xn @ w + b
        p = 1 / (1 + np.exp(-z))
        gw = Xn.T @ (p - y) / len(y)
        gb = (p - y).mean()
        w -= lr * gw; b -= lr * gb
    return w, b, X.mean(0), X.std(0) + 1e-8


def classify(docs, w, b, mu, sd):
    X = np.array([features(d) for d in docs])
    p = 1 / (1 + np.exp(-(((X - mu) / sd) @ w + b)))
    return p


# --------------------------------------------------------------- data plumbing
def build(seed=0):
    text = Path(HERE / "data/corpus.txt").read_text(encoding="utf-8")
    vocab_letters = "".join(sorted(set(c for c in text if c.isalpha() or c.isspace())))
    good = [text[i:i + DOC_LEN] for i in range(0, len(text) - DOC_LEN, DOC_LEN)]
    rng = np.random.default_rng(seed)
    rng.shuffle(good)
    n_val = int(0.15 * len(good))
    clean_val, good_train = good[:n_val], good[n_val:]

    kinds = ["noise", "mash", "repeat", "tagsoup"]
    junk = [make_junk(kinds[i % 4], DOC_LEN, rng, vocab_letters) for i in range(len(good_train))]

    mix = [(d, 1) for d in good_train] + [(d, 0) for d in junk]
    rng.shuffle(mix)
    docs = [d for d, _ in mix]; labels = np.array([l for _, l in mix])

    # train classifier on 40% of docs, evaluate on the rest
    idx = rng.permutation(len(docs)); split = int(0.4 * len(docs))
    tr, te = idx[:split], idx[split:]
    X = np.array([features(docs[i]) for i in tr])
    w, b, mu, sd = train_classifier(X, labels[tr])
    p_te = classify([docs[i] for i in te], w, b, mu, sd)
    pred = (p_te > 0.5).astype(int)
    truth = labels[te]
    tp = int(((pred == 1) & (truth == 1)).sum()); fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    acc = float((pred == truth).mean())

    # filtered corpus = documents the classifier keeps
    p_all = classify(docs, w, b, mu, sd)
    keep = [docs[i] for i in range(len(docs)) if p_all[i] > 0.5]
    stoi = {c: i for i, c in enumerate(sorted(set("\n".join(docs) + "\n".join(clean_val))))}
    return stoi, clean_val, docs, keep, dict(acc=acc, precision=precision, recall=recall,
                                             n_mix=len(docs), n_keep=len(keep),
                                             junk_frac_mix=float((labels == 0).mean()))


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
    model.eval(); tot = 0.0
    for _ in range(iters):
        x, y = batch(t, BATCH)
        _, loss = model(x, y); tot += loss.item()
    model.train(); return tot / iters


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
    stoi, clean_val, mix, kept, stats = build()
    vocab = len(stoi)
    val_t = encode(clean_val, stoi)
    mix_t, keep_t = encode(mix, stoi), encode(kept, stoi)
    print(f"classifier: acc {stats['acc']:.2f} | precision {stats['precision']:.2f} | "
          f"recall {stats['recall']:.2f}")
    print(f"corpus: mix {stats['n_mix']} docs ({100*stats['junk_frac_mix']:.0f}% junk) "
          f"-> filtered {stats['n_keep']} docs")

    results = {}
    for tag, tensor in [("unfiltered", mix_t), ("filtered", keep_t)]:
        model = train_on(tensor, vocab, tag)
        results[tag] = eval_loss(model, val_t)
        print(f"[{tag}] clean-val loss {results[tag]:.3f}")

    np.savez(CKPT / "results.npz", unfiltered=results["unfiltered"], filtered=results["filtered"],
             **{k: stats[k] for k in stats})
    print("wrote checkpoints/results.npz")


def plot():
    import plot_style as ps
    OUT.mkdir(parents=True, exist_ok=True)
    r = np.load(CKPT / "results.npz")
    fig, ax = ps.new_axes(6.4, 4.4)
    vals = [float(r["unfiltered"]), float(r["filtered"])]
    cols = [ps.SERIES[2], ps.SERIES[1]]
    bars = ax.bar(["unfiltered\n(50% junk)", "filtered\n(classifier kept)"], vals, color=cols, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9, color=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Quality filtering lowers loss on clean held-out text",
              "", "clean-val cross-entropy (nats/char)", OUT / "filter_effect.png")

    with open(OUT / "results.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"classifier_accuracy,{float(r['acc']):.3f}\n")
        f.write(f"classifier_precision,{float(r['precision']):.3f}\n")
        f.write(f"classifier_recall,{float(r['recall']):.3f}\n")
        f.write(f"unfiltered_clean_val,{float(r['unfiltered']):.3f}\n")
        f.write(f"filtered_clean_val,{float(r['filtered']):.3f}\n")
    print(f"wrote {OUT/'filter_effect.png'} + results.csv")


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
