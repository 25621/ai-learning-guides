"""Continued pretraining: keep teaching a model a new domain without erasing the
old one.

Most teams never pretrain from scratch — they take an existing model and adapt it
to their domain. We pretrain a small base model on English (tiny-shakespeare),
then *continue* pretraining it on a genuinely different domain — Python source
code from the standard library — and measure both sides of the ledger:

  * capability gained  — validation loss on held-out code
  * catastrophic forgetting — how much validation loss on English regresses

We also test the standard mitigation: mixing a little of the original data back
in ("replay"). All models share one character vocabulary so the two losses are
directly comparable.

    python continued_pretrain.py --run     # pretrain base, continue two ways, evaluate
    python continued_pretrain.py --plot     # figure from saved results

Reuses the GPT skeleton from project 08. The Python corpus is read from the local
standard library, so nothing is downloaded.
"""

import argparse
import copy
import glob
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
BLOCK = 128
BATCH = 32
BASE_STEPS = 500
CONT_STEPS = 400


def load_code(budget=1_200_000):
    """Concatenate standard-library .py files up to a character budget."""
    files = sorted(glob.glob("/usr/lib/python3*/*.py")) + \
        sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
    chunks, total = [], 0
    for fp in files:
        try:
            t = Path(fp).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not t.strip():
            continue
        chunks.append(t); total += len(t)
        if total >= budget:
            break
    return "\n".join(chunks)[:budget]


def build():
    english = Path(HERE / "data/corpus.txt").read_text(encoding="utf-8")
    code = load_code()
    # shared vocabulary over both domains
    stoi = {c: i for i, c in enumerate(sorted(set(english + code)))}

    def enc(txt):
        return torch.tensor([stoi[c] for c in txt if c in stoi], dtype=torch.long)

    e, c = enc(english), enc(code)
    ne, nc = int(0.9 * len(e)), int(0.9 * len(c))
    return dict(stoi=stoi,
                eng_train=e[:ne], eng_val=e[ne:],
                code_train=c[:nc], code_val=c[nc:])


def batch(t, bs):
    ix = torch.randint(len(t) - BLOCK - 1, (bs,))
    x = torch.stack([t[i:i + BLOCK] for i in ix])
    y = torch.stack([t[i + 1:i + 1 + BLOCK] for i in ix])
    return x, y


def mixed_batch(a, b, bs, frac_b):
    """A batch mixing corpus a and corpus b; frac_b of rows come from b."""
    nb = int(round(bs * frac_b))
    xa, ya = batch(a, bs - nb)
    xb, yb = batch(b, nb)
    return torch.cat([xa, xb]), torch.cat([ya, yb])


@torch.no_grad()
def eval_loss(model, t, iters=40):
    model.eval(); tot = 0.0
    for _ in range(iters):
        x, y = batch(t, BATCH); _, loss = model(x, y); tot += loss.item()
    model.train(); return tot / iters


def new_model(vocab):
    torch.manual_seed(0); torch.set_num_threads(12)
    cfg = Config(vocab_size=vocab, n_layer=4, n_head=4, n_embd=128, block_size=BLOCK)
    return GPT(cfg)


def train(model, sampler, steps, lr, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr, warmup=50)
        x, y = sampler()
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 150 == 0:
            print(f"  [{tag}] step {step}/{steps} loss {loss.item():.3f} "
                  f"({step/(time.time()-t0+1e-9):.2f} it/s)", flush=True)
    return model


def run():
    CKPT.mkdir(parents=True, exist_ok=True)
    d = build()
    vocab = len(d["stoi"])
    print(f"vocab {vocab} | english {len(d['eng_train'])/1e6:.2f}M tok | "
          f"code {len(d['code_train'])/1e6:.2f}M tok")

    # 1) base: pretrain on English only
    base = new_model(vocab)
    train(base, lambda: batch(d["eng_train"], BATCH), BASE_STEPS, 3e-3, "base/english")
    res = {"base": (eval_loss(base, d["eng_val"]), eval_loss(base, d["code_val"]))}
    print(f"[base] english {res['base'][0]:.3f} | code {res['base'][1]:.3f}")

    # 2) naive continued pretraining: 100% code
    naive = copy.deepcopy(base)
    train(naive, lambda: batch(d["code_train"], BATCH), CONT_STEPS, 1e-3, "continue/naive")
    res["naive"] = (eval_loss(naive, d["eng_val"]), eval_loss(naive, d["code_val"]))
    print(f"[naive] english {res['naive'][0]:.3f} | code {res['naive'][1]:.3f}")

    # 3) replay: 90% code + 10% English
    replay = copy.deepcopy(base)
    train(replay, lambda: mixed_batch(d["code_train"], d["eng_train"], BATCH, 0.1),
          CONT_STEPS, 1e-3, "continue/replay")
    res["replay"] = (eval_loss(replay, d["eng_val"]), eval_loss(replay, d["code_val"]))
    print(f"[replay] english {res['replay'][0]:.3f} | code {res['replay'][1]:.3f}")

    np.savez(CKPT / "results.npz",
             base=res["base"], naive=res["naive"], replay=res["replay"])
    print("wrote checkpoints/results.npz")


def plot():
    import plot_style as ps
    OUT.mkdir(parents=True, exist_ok=True)
    r = np.load(CKPT / "results.npz")
    models = ["base", "naive", "replay"]
    mlabel = ["base\n(English only)", "continued\n(100% code)", "continued\n(code + 10% replay)"]
    eng = [float(r[m][0]) for m in models]
    code = [float(r[m][1]) for m in models]

    fig, ax = ps.new_axes(7.4, 4.4)
    x = np.arange(len(models)); w = 0.36
    ax.bar(x - w / 2, eng, w, color=ps.SERIES[0], label="English val loss (old skill)")
    ax.bar(x + w / 2, code, w, color=ps.SERIES[1], label="code val loss (new skill)")
    for xi, v in zip(x - w / 2, eng):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=ps.INK_SECONDARY)
    for xi, v in zip(x + w / 2, code):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=ps.INK_SECONDARY)
    ax.set_xticks(x); ax.set_xticklabels(mlabel, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Continued pretraining: new skill vs. catastrophic forgetting",
              "", "validation loss (nats/char)", OUT / "continued_pretrain.png")

    with open(OUT / "results.csv", "w") as f:
        f.write("model,english_val_loss,code_val_loss\n")
        for m in models:
            f.write(f"{m},{float(r[m][0]):.3f},{float(r[m][1]):.3f}\n")
    print(f"wrote {OUT/'continued_pretrain.png'} + results.csv")


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
