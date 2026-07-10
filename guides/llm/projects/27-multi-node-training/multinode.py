"""Multi-node training, simulated on CPU ranks (gloo): what actually caps
throughput once a job spans machines.

A single laptop can't show real speed-up from "more nodes" — every rank shares the
same cores. What it *can* show honestly is the two things that decide multi-node
throughput in practice, and that a single GPU never has to worry about:

  1. communication cost — the all-reduce that averages gradients grows with the
     number of ranks, eating an ever-larger slice of each step;
  2. stragglers — synchronous training is a barrier, so one slow rank stalls
     everyone; the step takes as long as the *slowest* participant.

We run data-parallel DDP at world sizes 1/2/4, measure the compute vs communication
split, then inject an artificial straggler and watch the whole step wait for it.

    torchrun --nproc_per_node=1 multinode.py
    torchrun --nproc_per_node=2 multinode.py
    torchrun --nproc_per_node=4 multinode.py
    STRAGGLER_MS=150 torchrun --nproc_per_node=4 multinode.py
    python multinode.py --plot

Reuses the GPT skeleton from project 08.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT, CharData, cosine_lr   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
PER_RANK_BATCH = 16        # weak scaling: fixed work per rank
BLOCK = 128
TIMED_STEPS = 40


def allreduce_grads(model, world):
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad /= world


def run():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    straggler_ms = float(os.environ.get("STRAGGLER_MS", "0"))
    torch.manual_seed(0); torch.set_num_threads(max(1, 12 // world))

    text = Path(HERE / "data/corpus.txt").read_text(encoding="utf-8")
    data = CharData(text, block_size=BLOCK)
    cfg = Config(vocab_size=data.vocab_size, n_layer=4, n_head=4, n_embd=128, block_size=BLOCK)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))

    compute_t, comm_t = [], []
    for step in range(TIMED_STEPS + 5):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, TIMED_STEPS, 3e-3, warmup=5)
        x, y = data.batch("train", PER_RANK_BATCH)
        t0 = time.time()
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        if straggler_ms and rank == 0:                   # one slow node
            time.sleep(straggler_ms / 1000.0)
        t1 = time.time()
        allreduce_grads(model, world)                    # the synchronous barrier
        dist.barrier()
        t2 = time.time()
        opt.step()
        if step >= 5:                                    # skip warmup
            compute_t.append(t1 - t0); comm_t.append(t2 - t1)

    comp = float(np.mean(compute_t)); comm = float(np.mean(comm_t))
    # global throughput = all ranks' tokens / wall time per step
    step_time = comp + comm
    tokens_per_step = world * PER_RANK_BATCH * BLOCK
    thru = tokens_per_step / step_time
    if rank == 0:
        CKPT.mkdir(parents=True, exist_ok=True)
        tag = f"w{world}" + ("_straggler" if straggler_ms else "")
        np.savez(CKPT / f"scale_{tag}.npz", world=world, compute_ms=comp * 1e3,
                 comm_ms=comm * 1e3, tokens_per_s=thru, straggler_ms=straggler_ms)
        print(f"[{tag}] compute {comp*1e3:.0f} ms | comm {comm*1e3:.1f} ms "
              f"| comm frac {100*comm/step_time:.0f}% | {thru/1e3:.1f} k tok/s")
    dist.destroy_process_group()


def plot():
    import plot_style as ps
    OUT.mkdir(parents=True, exist_ok=True)
    worlds = [w for w in (1, 2, 4) if (CKPT / f"scale_w{w}.npz").exists()]
    data = {w: np.load(CKPT / f"scale_w{w}.npz") for w in worlds}

    # 1) compute vs communication per step
    fig, ax = ps.new_axes(7.2, 4.4)
    comp = [float(data[w]["compute_ms"]) for w in worlds]
    comm = [float(data[w]["comm_ms"]) for w in worlds]
    xs = np.arange(len(worlds))
    ax.bar(xs, comp, 0.55, color=ps.SERIES[0], label="compute")
    ax.bar(xs, comm, 0.55, bottom=comp, color=ps.SERIES[3], label="communication (all-reduce)")
    ax.set_xticks(xs); ax.set_xticklabels([f"{w} ranks" for w in worlds])
    for i, (c, m) in enumerate(zip(comp, comm)):
        ax.text(i, c + m + 1, f"{100*m/(c+m):.0f}% comm", ha="center", fontsize=8,
                color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Communication's share of a step grows with rank count",
              "", "per-step time (ms)", OUT / "comm_overhead.png")

    # 2) straggler effect at world=4
    if (CKPT / "scale_w4.npz").exists() and (CKPT / "scale_w4_straggler.npz").exists():
        base = np.load(CKPT / "scale_w4.npz"); strag = np.load(CKPT / "scale_w4_straggler.npz")
        fig, ax = ps.new_axes(6.6, 4.4)
        vals = [float(base["compute_ms"]) + float(base["comm_ms"]),
                float(strag["compute_ms"]) + float(strag["comm_ms"])]
        bars = ax.bar(["balanced\n(4 ranks)", f"one straggler\n(+{int(float(strag['straggler_ms']))} ms)"],
                      vals, color=[ps.SERIES[1], ps.SERIES[2]], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f} ms", ha="center",
                    fontsize=9, color=ps.INK_SECONDARY)
        ps.finish(fig, ax, "One straggler stalls the whole synchronous step",
                  "", "per-step time (ms)", OUT / "straggler.png")

    with open(OUT / "results.csv", "w") as f:
        f.write("world_size,compute_ms,comm_ms,comm_frac_pct,tokens_per_s\n")
        for w in worlds:
            d = data[w]; c, m = float(d["compute_ms"]), float(d["comm_ms"])
            f.write(f"{w},{c:.0f},{m:.1f},{100*m/(c+m):.0f},{float(d['tokens_per_s']):.0f}\n")
    print(f"wrote figures + results.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        plot()
    else:
        run()
