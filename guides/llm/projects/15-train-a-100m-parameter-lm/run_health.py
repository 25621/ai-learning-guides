"""The first run that feels real: hit a validation target, and learn to read
whether a run will get there.

The guide's milestone is a 100M-parameter model on OpenWebText pushed below 3.5
validation loss on an A100 — an 8-hour job. The hardware and the numbers change
with scale, but the skill does not: you must read a loss curve and judge, long
before it finishes, whether the run is *healthy*, *stalled*, or *diverging*, and
whether it will clear the number you set out to beat.

We train the same ~2.7M model three times, changing only the peak learning rate,
and race all three against a fixed validation target. Only the well-tuned run
gets there; the other two show exactly what stalled and diverging look like.

    python run_health.py --lr healthy      # 3e-3, warmup+cosine  (~4 min)
    python run_health.py --lr low          # 2e-4, too small       (~4 min)
    python run_health.py --lr high         # 1e-1, too large       (~4 min)
    python run_health.py --plot            # curve + target figure

Reuses the GPT skeleton from project 08. Curves go to checkpoints/ (gitignored);
--plot stitches the committed figure.
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
from model import Config, GPT, CharData, cosine_lr, estimate_loss   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
STEPS = 550
BATCH = 32
VAL_TARGET = 1.75                         # the char-level "number to beat"
RUNS = {"healthy": 3e-3, "low": 2e-4, "high": 1e-1}


def train(name):
    lr_peak = RUNS[name]
    text = Path(HERE / "data/corpus.txt").read_text(encoding="utf-8")
    data = CharData(text, block_size=128)
    torch.manual_seed(0); torch.set_num_threads(12)
    cfg = Config(vocab_size=data.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=128)
    model = GPT(cfg)
    print(f"[{name}] {model.num_params()/1e6:.2f}M params | peak lr {lr_peak:.0e}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr_peak, betas=(0.9, 0.95), weight_decay=0.1)
    curve, t0 = [], time.time()
    for step in range(STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, STEPS, lr_peak, warmup=60)
        if step % 25 == 0 or step == STEPS:
            losses = estimate_loss(model, data, BATCH)
            v = losses["val"]
            curve.append((step, v))
            if step % 100 == 0 or step == STEPS:
                print(f"  [{name}] step {step}/{STEPS} | val {v:.3f} | "
                      f"{step/(time.time()-t0+1e-9):.2f} it/s", flush=True)
        if step == STEPS:
            break
        x, y = data.batch("train", BATCH)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    CKPT.mkdir(parents=True, exist_ok=True)
    curve = np.array(curve)
    np.savez(CKPT / f"curve_{name}.npz", curve=curve)
    crossed = np.where(curve[:, 1] < VAL_TARGET)[0]
    cross_step = int(curve[crossed[0], 0]) if len(crossed) else -1
    print(f"[{name}] final val {curve[-1, 1]:.3f} | "
          f"{'crossed target at step '+str(cross_step) if cross_step >= 0 else 'never crossed target'}")


def plot():
    import plot_style as ps
    OUT.mkdir(parents=True, exist_ok=True)
    order = ["healthy", "low", "high"]
    label = {"healthy": "healthy (lr 3e-3)", "low": "stalled (lr 2e-4)",
             "high": "diverging (lr 1e-1)"}
    color = {"healthy": ps.SERIES[1], "low": ps.SERIES[3], "high": ps.SERIES[2]}
    names = [n for n in order if (CKPT / f"curve_{n}.npz").exists()]

    fig, ax = ps.new_axes(7.2, 4.4)
    for n in names:
        c = np.load(CKPT / f"curve_{n}.npz")["curve"]
        ax.plot(c[:, 0], np.clip(c[:, 1], 0, 4.5), color=color[n], label=label[n])
    ax.axhline(VAL_TARGET, color=ps.INK_MUTED, ls="--", lw=1)
    ax.text(STEPS * 0.55, VAL_TARGET + 0.05, f"target to beat = {VAL_TARGET}",
            color=ps.INK_MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Healthy, stalled, diverging — reading a run against a target",
              "training step", "validation loss (nats/char)", OUT / "run_health.png")

    with open(OUT / "results.csv", "w") as f:
        f.write("run,peak_lr,final_val_loss,beat_target\n")
        for n in names:
            c = np.load(CKPT / f"curve_{n}.npz")["curve"]
            f.write(f"{n},{RUNS[n]:.0e},{float(c[-1, 1]):.3f},{bool(c[-1, 1] < VAL_TARGET)}\n")
    print(f"wrote {OUT/'run_health.png'} + results.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", choices=list(RUNS))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.lr:
        train(args.lr)
    if args.plot:
        plot()


if __name__ == "__main__":
    main()
