"""FSDP from scratch, by hand, on CPU ranks (gloo) — the magic in ~40 lines.

FSDP (a.k.a. ZeRO-3) shards a model's parameters, gradients, and optimizer state
across ranks so no single device holds the whole thing. Each step it
*all-gathers* the full weights to compute, then *reduce-scatters* the gradients so
each rank ends up owning — and updating — only its shard.

We implement exactly that for the whole model at once (a toy; real FSDP shards
layer-by-layer to keep the gathered copy small). Alongside the sharded optimizer we
keep a *replicated* one (plain data-parallel: the whole model + optimizer state on
every rank) fed the identical averaged gradient, and check every step that the two
agree — because AdamW is elementwise, sharding only *partitions* the same update, so
they stay bitwise identical.

    torchrun --nproc_per_node=2 fsdp_toy.py     # shard across 2 ranks
    python fsdp_toy.py --plot

Reuses the GPT skeleton from project 08.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT, CharData, cosine_lr   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
STEPS = 300
GLOBAL_BATCH = 32
BLOCK = 128


def train_sharded():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    # fixed thread count across world sizes so the only difference is the sharding math
    torch.manual_seed(0); torch.set_num_threads(4)

    text = Path(HERE / "data/corpus.txt").read_text(encoding="utf-8")
    data = CharData(text, block_size=BLOCK)
    cfg = Config(vocab_size=data.vocab_size, n_layer=4, n_head=4, n_embd=128, block_size=BLOCK)
    model = GPT(cfg)                                      # every rank builds the same model

    # --- shard the flat parameter vector into `world` equal, padded pieces ---
    flat = parameters_to_vector(model.parameters()).detach()
    P = flat.numel()
    shard_len = (P + world - 1) // world
    pad = shard_len * world - P
    full = torch.cat([flat, torch.zeros(pad)]) if pad else flat.clone()
    master = full[rank * shard_len:(rank + 1) * shard_len].clone()   # THIS rank's owned shard
    master.requires_grad_(False)
    opt = torch.optim.AdamW([master], lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)

    # A full REPLICATED reference (plain data-parallel: every rank keeps the whole
    # model + optimizer state) — used only to prove the sharded path is identical.
    repl = full.clone(); repl.requires_grad_(False)
    opt_repl = torch.optim.AdamW([repl], lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)

    my_slice = slice(rank * shard_len, (rank + 1) * shard_len)
    param_bytes = master.numel() * master.element_size()
    if rank == 0:
        print(f"world={world} | {P/1e6:.3f}M params total | sharded: each rank owns "
              f"{master.numel()/1e6:.3f}M ({param_bytes/1e6:.2f} MB) | replicated: full "
              f"{P/1e6:.3f}M ({P*4/1e6:.2f} MB) per rank")

    per_rank_batch = GLOBAL_BATCH // world
    curve, divergence = [], []
    for step in range(STEPS + 1):
        # 1) ALL-GATHER: reconstruct the full parameters from every rank's shard
        gathered = [torch.empty(shard_len) for _ in range(world)]
        dist.all_gather(gathered, master)
        full_params = torch.cat(gathered)[:P]
        vector_to_parameters(full_params, model.parameters())

        for g in list(opt.param_groups) + list(opt_repl.param_groups):
            g["lr"] = cosine_lr(step, STEPS, 3e-3, warmup=40)

        # deterministic global batch, split across ranks (data parallelism)
        torch.manual_seed(1000 + step)
        gx, gy = data.batch("train", GLOBAL_BATCH)
        x = gx[rank * per_rank_batch:(rank + 1) * per_rank_batch]
        y = gy[rank * per_rank_batch:(rank + 1) * per_rank_batch]
        _, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()

        # every rank's reconstructed weights should equal the replicated reference
        div = float((full_params - repl[:P]).abs().max())
        if step % 25 == 0 or step == STEPS:
            lt = loss.detach().clone(); dist.all_reduce(lt, op=dist.ReduceOp.SUM)
            if rank == 0:
                curve.append((step, (lt / world).item())); divergence.append((step, div))

        if step == STEPS:
            break

        # 2) REDUCE the gradients across ranks (data-parallel average)
        gvec = parameters_to_vector([p.grad for p in model.parameters()]).detach()
        gfull = torch.cat([gvec, torch.zeros(pad)]) if pad else gvec.clone()
        dist.all_reduce(gfull, op=dist.ReduceOp.SUM); gfull /= world

        # 3a) SHARDED update: each rank owns and updates only its slice
        master.grad = gfull[my_slice].clone(); opt.step()
        # 3b) REPLICATED update: same gradient, whole model (the thing we match)
        repl.grad = gfull.clone(); opt_repl.step()

    if rank == 0:
        CKPT.mkdir(parents=True, exist_ok=True)
        np.savez(CKPT / "fsdp.npz", curve=np.array(curve), divergence=np.array(divergence),
                 total_params=P, shard_params=master.numel(), shard_mb=param_bytes / 1e6,
                 repl_mb=P * 4 / 1e6, world=world)
        print(f"[world={world}] final loss {curve[-1][1]:.4f} | "
              f"max sharded-vs-replicated divergence {max(d for _, d in divergence):.2e}")
    dist.destroy_process_group()


def plot():
    import plot_style as ps
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(CKPT / "fsdp.npz")
    curve, div = d["curve"], d["divergence"]
    world = int(d["world"]); maxdiff = float(np.max(div[:, 1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4), dpi=110)
    for ax in (ax1, ax2):
        ax.set_facecolor(ps.SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(ps.BASELINE)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=9)
        ax.grid(True, color=ps.GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.patch.set_facecolor(ps.SURFACE)

    ax1.plot(curve[:, 0], curve[:, 1], color=ps.SERIES[1])
    ax1.set_title(f"Sharded {world}-rank training loss", color=ps.INK, fontsize=11,
                  loc="left", pad=10)
    ax1.set_xlabel("training step", color=ps.INK_SECONDARY, fontsize=10)
    ax1.set_ylabel("training loss (nats/char)", color=ps.INK_SECONDARY, fontsize=10)

    ax2.semilogy(div[:, 0], np.clip(div[:, 1], 1e-12, None), color=ps.SERIES[2])
    ax2.set_title("Sharded vs replicated: bitwise identical", color=ps.INK, fontsize=11,
                  loc="left", pad=10)
    ax2.set_xlabel("training step", color=ps.INK_SECONDARY, fontsize=10)
    ax2.set_ylabel("max weight difference", color=ps.INK_SECONDARY, fontsize=10)
    ax2.text(0.04, 0.9, f"max divergence = {maxdiff:.1e}", transform=ax2.transAxes,
             color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fsdp_equivalence.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)

    with open(OUT / "results.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"world_size,{world}\n")
        f.write(f"total_params_millions,{float(d['total_params'])/1e6:.3f}\n")
        f.write(f"sharded_master_mb_per_rank,{float(d['shard_mb']):.2f}\n")
        f.write(f"replicated_master_mb_per_rank,{float(d['repl_mb']):.2f}\n")
        f.write(f"final_loss,{float(curve[-1,1]):.4f}\n")
        f.write(f"max_sharded_vs_replicated_divergence,{maxdiff:.2e}\n")
    print(f"wrote {OUT/'fsdp_equivalence.png'} + results.csv (max divergence {maxdiff:.1e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        plot()
    else:
        train_sharded()
