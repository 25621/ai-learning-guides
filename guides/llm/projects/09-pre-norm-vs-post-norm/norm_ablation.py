"""Pre-norm vs post-norm: where you put the normalization decides whether a deep
transformer trains robustly or falls apart.

Pre-norm (`x = x + f(Norm(x))`) keeps a clean residual highway, so gradients reach
every layer and training is forgiving of the learning rate. Post-norm (the 2017
design, `x = Norm(x + f(x))`) compounds scale through the stack; past a modest
depth its gradient signal is suppressed and it stalls unless everything is tuned
just right. We demonstrate the robustness gap two ways:

  1. a learning-rate sweep at fixed depth — pre-norm stays low everywhere,
     post-norm only works in a narrow low-LR band, then stalls;
  2. loss curves at an aggressive LR — pre-norm descends, post-norm flat-lines.

    python norm_ablation.py --corpus data/corpus.txt      # ~7 min on CPU

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
from model import Config, GPT, CharData, cosine_lr, estimate_loss    # noqa: E402

OUT = HERE / "outputs"
DEPTH = 8
LRS = [1.5e-3, 4e-3, 8e-3, 1.4e-2]


def train_one(data, norm, lr, steps, batch, record=False, device="cpu"):
    torch.manual_seed(0)                                 # identical init across runs
    cfg = Config(vocab_size=data.vocab_size, n_layer=DEPTH, n_head=6, n_embd=96,
                 block_size=64, norm=norm)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    curve = []
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr, warmup=0)
        x, y = data.batch("train", batch, device)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if record and step % 10 == 0:
            curve.append((step, loss.item()))
    final = estimate_loss(model, data, batch, iters=20)["val"]
    return final, np.array(curve) if record else None


def plot_sweep(pre_final, post_final, path):
    import plot_style as ps
    fig, ax = ps.new_axes(7.2, 4.4)
    ax.plot(LRS, pre_final, "-o", color=ps.SERIES[0], label="pre-norm", linewidth=1.8)
    ax.plot(LRS, post_final, "-o", color=ps.SERIES[2], label="post-norm", linewidth=1.8)
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=10)
    ps.finish(fig, ax, f"Pre-norm trains at every learning rate; post-norm ({DEPTH} layers) is fragile",
              "learning rate (log)", "final validation loss", path)


def plot_curves(pre_curve, post_curve, lr, path):
    import plot_style as ps
    fig, ax = ps.new_axes(7.2, 4.2)
    ax.plot(pre_curve[:, 0], pre_curve[:, 1], color=ps.SERIES[0], label="pre-norm", linewidth=1.7)
    ax.plot(post_curve[:, 0], post_curve[:, 1], color=ps.SERIES[2], label="post-norm", linewidth=1.7)
    ax.legend(frameon=False, fontsize=10)
    ps.finish(fig, ax, f"At lr={lr:.0e}, pre-norm descends while post-norm stalls",
              "training step", "training loss", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "data/corpus.txt"))
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(12)

    text = Path(args.corpus).read_text(encoding="utf-8")
    data = CharData(text, block_size=64)
    t0 = time.time()
    pre_final, post_final = [], []
    pre_curve = post_curve = None
    for lr in LRS:
        rec = lr == LRS[-1]                              # record curves at the top LR
        pf, pc = train_one(data, "pre", lr, args.steps, args.batch, record=rec)
        qf, qc = train_one(data, "post", lr, args.steps, args.batch, record=rec)
        pre_final.append(pf); post_final.append(qf)
        if rec:
            pre_curve, post_curve = pc, qc
        print(f"  lr={lr:.1e} | pre-norm {pf:.3f} | post-norm {qf:.3f} | "
              f"{time.time()-t0:.0f}s", flush=True)

    plot_sweep(pre_final, post_final, OUT / "lr_robustness.png")
    plot_curves(pre_curve, post_curve, LRS[-1], OUT / "loss_curves.png")
    lines = ["learning_rate,pre_norm_val,post_norm_val"]
    for lr, pf, qf in zip(LRS, pre_final, post_final):
        lines.append(f"{lr:.1e},{pf:.3f},{qf:.3f}")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")
    print("wrote figures + results.csv\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
