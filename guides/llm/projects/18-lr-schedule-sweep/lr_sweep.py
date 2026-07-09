"""LR schedule sweep: same model, same data, three ways to move one number.

The learning rate is a single scalar that changes over training, and how you
change it quietly decides where you land. We train the identical ~1.9M model
three times — cosine decay, Warmup-Stable-Decay (WSD), and constant — and compare
the schedules side by side with the validation loss each produces.

    python lr_sweep.py --sched cosine       # ~3 min
    python lr_sweep.py --sched wsd          # ~3 min
    python lr_sweep.py --sched constant     # ~3 min
    python lr_sweep.py --plot               # schedule + val-loss figures

Reuses the GPT skeleton from project 08. Curves are written to checkpoints/
(gitignored); --plot stitches them into the committed figures.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT, CharData, estimate_loss   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
STEPS = 600
BATCH = 32
PEAK = 3e-3
WARMUP = 60
MIN_FRAC = 0.1


def lr_at(step, sched):
    if step < WARMUP:                                    # shared linear warmup
        return PEAK * step / WARMUP
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)      # 0..1 after warmup
    if sched == "constant":
        return PEAK
    if sched == "cosine":
        return PEAK * (MIN_FRAC + (1 - MIN_FRAC) * 0.5 * (1 + math.cos(math.pi * prog)))
    if sched == "wsd":                                   # stable, then short decay tail
        decay_frac = 0.2
        if prog < 1 - decay_frac:
            return PEAK
        d = (prog - (1 - decay_frac)) / decay_frac       # 0..1 across the tail
        return PEAK * (MIN_FRAC + (1 - MIN_FRAC) * (1 - d))
    raise ValueError(sched)


def train(sched):
    text = Path(HERE / "data/corpus.txt").read_text(encoding="utf-8")
    data = CharData(text, block_size=128)
    torch.manual_seed(0); torch.set_num_threads(12)
    cfg = Config(vocab_size=data.vocab_size, n_layer=4, n_head=4, n_embd=128, block_size=128)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK, betas=(0.9, 0.95), weight_decay=0.1)
    curve, lrs, t0 = [], [], time.time()
    for step in range(STEPS + 1):
        lr = lr_at(step, sched)
        for g in opt.param_groups:
            g["lr"] = lr
        lrs.append(lr)
        if step % 50 == 0 or step == STEPS:
            losses = estimate_loss(model, data, BATCH)
            curve.append((step, losses["train"], losses["val"]))
            print(f"  [{sched}] step {step}/{STEPS} | val {losses['val']:.3f} | lr {lr:.2e} | "
                  f"{step/(time.time()-t0+1e-9):.2f} it/s", flush=True)
        if step == STEPS:
            break
        x, y = data.batch("train", BATCH)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    CKPT.mkdir(parents=True, exist_ok=True)
    np.savez(CKPT / f"curve_{sched}.npz", curve=np.array(curve), lrs=np.array(lrs))
    print(f"[{sched}] final val {curve[-1][2]:.3f}")


def plot():
    import plot_style as ps
    OUT.mkdir(parents=True, exist_ok=True)
    order = ["cosine", "wsd", "constant"]
    names = [s for s in order if (CKPT / f"curve_{s}.npz").exists()]
    label = {"cosine": "cosine decay", "wsd": "WSD", "constant": "constant"}

    fig, ax = ps.new_axes(7.2, 3.6)
    for i, s in enumerate(names):
        lrs = np.load(CKPT / f"curve_{s}.npz")["lrs"]
        ax.plot(np.arange(len(lrs)), lrs, color=ps.SERIES[i], label=label[s])
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Three learning-rate schedules", "training step",
              "learning rate", OUT / "schedules.png")

    fig, ax = ps.new_axes(7.2, 4.2)
    for i, s in enumerate(names):
        c = np.load(CKPT / f"curve_{s}.npz")["curve"]
        ax.plot(c[:, 0], c[:, 2], color=ps.SERIES[i], label=label[s])
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Validation loss under each schedule", "training step",
              "validation loss (nats/char)", OUT / "val_loss.png")

    with open(OUT / "results.csv", "w") as f:
        f.write("schedule,final_val_loss\n")
        for s in names:
            c = np.load(CKPT / f"curve_{s}.npz")["curve"]
            f.write(f"{s},{float(c[-1, 2]):.3f}\n")
    print(f"wrote {OUT/'schedules.png'}, val_loss.png, results.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sched", choices=["cosine", "wsd", "constant"])
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.sched:
        train(args.sched)
    if args.plot:
        plot()


if __name__ == "__main__":
    main()
