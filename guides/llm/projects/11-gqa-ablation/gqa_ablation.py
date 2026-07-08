"""GQA ablation: measure the KV-cache saving against the validation-loss cost.

Multi-head attention gives every query head its own keys and values; MQA shares a
single KV head across all of them; GQA sits in between. Fewer KV heads shrink the
KV cache (the serving-memory bottleneck) but can cost quality. We train three
otherwise-identical models — MHA, GQA-2, MQA — on the same data and plot the
cache/quality trade-off directly.

    python gqa_ablation.py --corpus data/corpus.txt --config mha   # one at a time
    python gqa_ablation.py --config gqa2
    python gqa_ablation.py --config mqa
    python gqa_ablation.py --plot

Reuses the GPT skeleton from project 08.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT, CharData, train_model, estimate_loss   # noqa: E402

OUT = HERE / "outputs"
N_HEAD = 8
CONFIGS = {                       # name: (n_kv_heads, label, color_idx)
    "mha": (8, "MHA (8 kv)", 0),
    "gqa2": (2, "GQA-2", 1),
    "mqa": (1, "MQA (1 kv)", 2),
}
# a realistic serving config, to report cache sizes that mean something
SERVE = dict(d=4096, n_head=32, n_layers=32, seq=4096, dtype_bytes=2)


def kv_cache_gb(n_kv_heads_frac):
    d_head = SERVE["d"] // SERVE["n_head"]
    n_kv = max(1, round(SERVE["n_head"] * n_kv_heads_frac))
    return 2 * n_kv * d_head * SERVE["seq"] * SERVE["n_layers"] * SERVE["dtype_bytes"] / 1e9


def run_config(key, data, steps, batch):
    nkv, label, _ = CONFIGS[key]
    torch.manual_seed(0)
    cfg = Config(vocab_size=data.vocab_size, n_layer=4, n_head=N_HEAD, n_kv_heads=nkv,
                 n_embd=192, block_size=128)
    model = GPT(cfg)
    print(f"[{key}] {label}: {model.num_params()/1e6:.2f}M params, {nkv}/{N_HEAD} kv heads")
    curve = train_model(model, data, steps=steps, batch_size=batch, lr=3e-3,
                        warmup=100, eval_every=200, tag=f"{key} ")
    final = estimate_loss(model, data, batch, iters=40)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"_{key}.npz", val=final["val"], train=final["train"],
             nkv=nkv, curve=np.array(curve), cache_gb=kv_cache_gb(nkv / N_HEAD))
    print(f"[{key}] final val {final['val']:.3f} | serving KV cache {kv_cache_gb(nkv/N_HEAD):.2f} GB")


def make_plots():
    import plot_style as ps
    data = {k: np.load(OUT / f"_{k}.npz") for k in CONFIGS}
    # trade-off scatter: cache (x) vs val loss (y)
    fig, ax = ps.new_axes(7.0, 4.4)
    for k in CONFIGS:
        d = data[k]; _, label, ci = CONFIGS[k]
        ax.scatter(float(d["cache_gb"]), float(d["val"]), s=120, color=ps.SERIES[ci], zorder=3)
        ax.annotate(label, (float(d["cache_gb"]), float(d["val"])),
                    textcoords="offset points", xytext=(8, 6), fontsize=9, color=ps.INK_SECONDARY)
    ps.finish(fig, ax, "The GQA trade-off: KV cache vs. quality",
              "serving KV cache (GB, 4k ctx)", "validation loss", OUT / "tradeoff.png")

    # loss curves
    fig, ax = ps.new_axes(7.2, 4.2)
    for k in CONFIGS:
        c = data[k]["curve"]; _, label, ci = CONFIGS[k]
        ax.plot(c[:, 0], c[:, 2], color=ps.SERIES[ci], label=label, linewidth=1.6)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Validation loss during training (MHA / GQA / MQA)",
              "training step", "validation loss", OUT / "loss_curves.png")

    lines = ["config,n_kv_heads,val_loss,serving_kv_cache_GB,cache_vs_mha"]
    base = float(data["mha"]["cache_gb"])
    for k in CONFIGS:
        d = data[k]
        lines.append(f"{CONFIGS[k][1]},{int(d['nkv'])},{float(d['val']):.3f},"
                     f"{float(d['cache_gb']):.2f},{base/float(d['cache_gb']):.0f}x")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")
    print("wrote figures + results.csv\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "data/corpus.txt"))
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--steps", type=int, default=800)
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
        raise SystemExit("pass --config mha|gqa2|mqa or --plot")


if __name__ == "__main__":
    main()
