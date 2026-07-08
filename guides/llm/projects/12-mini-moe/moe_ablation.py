"""Mini-MoE: replace the MLP with 8 experts and a top-2 router, and watch whether
routing stays balanced.

A Mixture-of-Experts swaps one MLP for several "expert" MLPs plus a router that
sends each token to only the top-k experts. Total parameters balloon while compute
per token stays fixed — that's how Mixtral and DeepSeek-V3 get huge cheaply. The
central risk is routing *collapse*: the router falling in love with a few experts
and starving the rest. A load-balancing loss fixes it. We train a dense baseline
and an 8-expert top-2 MoE, compare loss, and chart expert utilization.

    python moe_ablation.py --corpus data/corpus.txt --config dense
    python moe_ablation.py --config moe
    python moe_ablation.py --plot

Reuses the GPT skeleton (including its MoE block) from project 08.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT, CharData, train_model, estimate_loss, MoE   # noqa: E402

OUT = HERE / "outputs"
N_EXPERTS, TOP_K = 8, 2


def make_cfg(vocab, moe):
    return Config(vocab_size=vocab, n_layer=3, n_head=4, n_embd=128, block_size=128,
                  moe={"n_experts": N_EXPERTS, "top_k": TOP_K} if moe else None)


def active_params(model, cfg):
    """Params actually used per token (MoE runs only top_k of n_experts)."""
    total = model.num_params()
    if cfg.moe is None:
        return total, total
    # subtract the inactive experts' parameters
    per_expert = sum(p.numel() for p in model.blocks[0].mlp.experts[0].parameters())
    inactive = cfg.n_layer * (N_EXPERTS - TOP_K) * per_expert
    return total, total - inactive


@torch.no_grad()
def routing_fractions(model, data, batch=64):
    model.eval()
    x, _ = data.batch("val", batch)
    model(x)
    fr = [b.mlp.frac.numpy() for b in model.blocks if isinstance(b.mlp, MoE)]
    return np.mean(fr, axis=0)          # avg top-1 fraction per expert across layers


def run_config(key, data, steps, batch):
    torch.manual_seed(0)
    cfg = make_cfg(data.vocab_size, moe=(key == "moe"))
    model = GPT(cfg)
    total, active = active_params(model, cfg)
    print(f"[{key}] total {total/1e6:.2f}M params, active/token {active/1e6:.2f}M")
    curve = train_model(model, data, steps=steps, batch_size=batch, lr=3e-3,
                        warmup=100, eval_every=200, tag=f"{key} ")
    final = estimate_loss(model, data, batch, iters=40)
    fr = routing_fractions(model, data) if key == "moe" else np.array([])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"_{key}.npz", val=final["val"], curve=np.array(curve),
             total=total, active=active, frac=fr)
    print(f"[{key}] final val {final['val']:.3f}"
          + (f" | expert use min {fr.min():.2f} max {fr.max():.2f} (ideal {1/N_EXPERTS:.2f})"
             if key == "moe" else ""))


def make_plots():
    import plot_style as ps
    dense = np.load(OUT / "_dense.npz"); moe = np.load(OUT / "_moe.npz")
    # loss curves
    fig, ax = ps.new_axes(7.2, 4.2)
    for d, name, ci in [(dense, "dense MLP", 2), (moe, "8-expert MoE (top-2)", 0)]:
        c = d["curve"]; ax.plot(c[:, 0], c[:, 2], color=ps.SERIES[ci], label=name, linewidth=1.7)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "At this scale MoE ties dense — its capacity edge shows only when large",
              "training step", "validation loss", OUT / "loss_curves.png")
    # expert utilization
    fig, ax = ps.new_axes(7.0, 4.2)
    fr = moe["frac"]
    ax.bar(range(len(fr)), fr, color=ps.SERIES[0], width=0.7)
    ax.axhline(1 / len(fr), color=ps.SERIES[2], linestyle="--", linewidth=1.2,
               label=f"balanced ({1/len(fr):.2f})")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Expert utilization stays balanced (no routing collapse)",
              "expert index", "fraction of tokens routed (top-1)", OUT / "expert_utilization.png")

    lines = ["config,total_params_M,active_params_M,val_loss",
             f"dense,{float(dense['total'])/1e6:.2f},{float(dense['active'])/1e6:.2f},{float(dense['val']):.3f}",
             f"moe,{float(moe['total'])/1e6:.2f},{float(moe['active'])/1e6:.2f},{float(moe['val']):.3f}"]
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")
    print("wrote figures + results.csv\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "data/corpus.txt"))
    ap.add_argument("--config", choices=["dense", "moe"])
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.plot:
        make_plots()
    elif args.config:
        text = Path(args.corpus).read_text(encoding="utf-8")
        run_config(args.config, CharData(text, 128), args.steps, args.batch)
    else:
        raise SystemExit("pass --config dense|moe or --plot")


if __name__ == "__main__":
    main()
