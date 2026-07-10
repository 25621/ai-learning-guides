"""Reproduce the Chinchilla result in miniature: train a handful of model sizes
and watch the compute-optimal frontier draw itself.

We use Chinchilla's "Approach 1" (training curves): give each of several model
sizes the *same wall-clock budget*, and log validation loss against cumulative
compute (FLOPs = 6 x N x tokens_seen). Two truths fall out:

  * on the FLOP axis the size curves CROSS — a small model wins at low compute, a
    bigger model overtakes it as the budget grows;
  * slicing the curves at fixed FLOP budgets (iso-FLOP) gives U-shaped loss-vs-size
    curves whose minima march to larger models as compute increases — the
    Chinchilla "optimal size grows with compute" result.

    python chinchilla.py --run --budget-sec 100     # ~9 min on CPU (5 sizes)
    python chinchilla.py --plot                      # the two figures

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
from model import Config, GPT, CharData, cosine_lr, estimate_loss   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
BATCH, BLOCK = 16, 128

# (name, n_layer, n_head, n_embd)
SIZES = [
    ("d32",  4, 4, 32),
    ("d64",  4, 4, 64),
    ("d96",  4, 6, 96),
    ("d128", 4, 8, 128),
    ("d192", 6, 6, 192),
]


def train_one(name, L, H, D, data, budget_sec):
    torch.manual_seed(0); torch.set_num_threads(12)
    cfg = Config(vocab_size=data.vocab_size, n_layer=L, n_head=H, n_embd=D, block_size=BLOCK)
    model = GPT(cfg)
    N = model.num_params()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    log = []                                             # (tokens, flops, val_loss)
    t0 = time.time(); step = 0; tokens = 0
    # rough total-step guess for the cosine schedule (updated as we go is overkill)
    approx_total = max(200, int(budget_sec * 3))
    while time.time() - t0 < budget_sec:
        if step % 25 == 0:
            vl = estimate_loss(model, data, BATCH, iters=10)["val"]
            log.append((tokens, 6 * N * tokens, vl))
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, approx_total, 3e-3, warmup=40)
        x, y = data.batch("train", BATCH)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1; tokens += BATCH * BLOCK
    vl = estimate_loss(model, data, BATCH, iters=10)["val"]
    log.append((tokens, 6 * N * tokens, vl))
    print(f"[{name}] {N/1e6:.3f}M params | {step} steps | {tokens/1e6:.2f}M tokens | "
          f"final val {vl:.3f}")
    return N, np.array(log)


def run(budget_sec):
    CKPT.mkdir(parents=True, exist_ok=True)
    text = Path(HERE / "data/corpus.txt").read_text(encoding="utf-8")
    data = CharData(text, block_size=BLOCK)
    params, logs = [], {}
    for name, L, H, D in SIZES:
        N, log = train_one(name, L, H, D, data, budget_sec)
        params.append((name, N)); logs[name] = log
    np.savez(CKPT / "curves.npz",
             names=[n for n, _ in params], nparams=[N for _, N in params],
             **{f"log_{n}": logs[n] for n, _ in params})
    print("wrote checkpoints/curves.npz")


def _interp_loss(log, flop_budget):
    """Loss of a model at a given cumulative-FLOP budget (log-linear interp)."""
    f, l = log[:, 1], log[:, 2]
    if flop_budget < f[1] or flop_budget > f[-1]:
        return None
    return float(np.interp(np.log(flop_budget), np.log(f[1:]), l[1:]))


def plot():
    import plot_style as ps
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(CKPT / "curves.npz", allow_pickle=True)
    names = list(d["names"]); nparams = list(d["nparams"])
    logs = {n: d[f"log_{n}"] for n in names}

    # 1) loss vs compute — the crossing training curves
    fig, ax = ps.new_axes(7.4, 4.6)
    for i, n in enumerate(names):
        lg = logs[n]
        ax.plot(lg[1:, 1], lg[1:, 2], color=ps.SERIES[i % len(ps.SERIES)],
                label=f"{n} ({nparams[i]/1e6:.2f}M)")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8, title="model size")
    ps.finish(fig, ax, "Loss vs compute: bigger models overtake smaller ones",
              "training compute  (FLOPs = 6ND, log scale)", "validation loss (nats/char)",
              OUT / "loss_vs_compute.png")

    # 2) iso-FLOP slices — U-curves whose minima move right with compute
    all_f = np.concatenate([logs[n][1:, 1] for n in names])
    lo, hi = np.percentile(all_f, 25), np.percentile(all_f, 92)
    budgets = np.exp(np.linspace(np.log(lo), np.log(hi), 3))
    fig, ax = ps.new_axes(7.4, 4.6)
    opt_pts = []
    for j, C in enumerate(budgets):
        xs, ys = [], []
        for i, n in enumerate(names):
            v = _interp_loss(logs[n], C)
            if v is not None:
                xs.append(nparams[i]); ys.append(v)
        if len(xs) >= 3:
            ax.plot(xs, ys, "-o", color=ps.SERIES[j], label=f"{C:.1e} FLOP")
            k = int(np.argmin(ys)); opt_pts.append((xs[k], ys[k]))
    for x, y in opt_pts:
        ax.plot([x], [y], marker="*", ms=15, color=ps.INK, zorder=5)
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8, title="compute budget")
    ps.finish(fig, ax, "Iso-FLOP curves: the optimal size grows with compute (★)",
              "model size  (non-embedding params, log scale)", "validation loss (nats/char)",
              OUT / "iso_flop.png")

    with open(OUT / "results.csv", "w") as f:
        f.write("size,params_millions,final_val_loss,final_flops\n")
        for i, n in enumerate(names):
            lg = logs[n]
            f.write(f"{n},{nparams[i]/1e6:.3f},{lg[-1,2]:.3f},{lg[-1,1]:.2e}\n")
        f.write("\ncompute_budget_flops,optimal_size_params_millions,loss\n")
        for (x, y), C in zip(opt_pts, budgets):
            f.write(f"{C:.2e},{x/1e6:.3f},{y:.3f}\n")
    print(f"wrote {OUT/'loss_vs_compute.png'}, iso_flop.png, results.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--budget-sec", type=float, default=100)
    args = ap.parse_args()
    if args.run:
        run(args.budget_sec)
    if args.plot:
        plot()


if __name__ == "__main__":
    main()
