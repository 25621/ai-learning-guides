"""Long-context extension: stretch a short-context model to longer text by
rescaling how RoPE counts position (position interpolation), then briefly
fine-tuning.

A model trained with a 128-token window has never seen the RoPE rotation angles
that later positions produce, so feeding it a long sequence makes its predictions
degrade past the training length. Position Interpolation (the idea behind YaRN)
*compresses* the position indices back into the trained range — but that also
squashes every relative distance, so the model needs a short fine-tune to adapt.
We measure per-position loss (a needle-in-a-haystack proxy: can the model still
predict deep into the sequence?) for naive extension vs interpolation-plus-finetune.

    python long_context.py --corpus data/corpus.txt      # ~4 min on CPU

Loads the trained checkpoint from project 08 and fine-tunes it briefly.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import GPT, CharData, cosine_lr    # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE.parent / "08-nanogpt-reproduction" / "checkpoints" / "gpt.pt"


@torch.no_grad()
def per_position_loss(model, data, L, n_seqs, scale):
    model.eval(); model.rope_scale = scale
    d = data.val
    total = torch.zeros(L)
    starts = torch.linspace(0, len(d) - L - 2, n_seqs).long()
    for s in starts:
        x = d[s:s + L].unsqueeze(0)
        y = d[s + 1:s + L + 1].unsqueeze(0)
        logits, _ = model(x)
        total += F.cross_entropy(logits[0], y[0], reduction="none")
    return (total / n_seqs).numpy()


def finetune_at_length(model, data, L, scale, steps, batch, lr=5e-4):
    """A short fine-tune at the long context with interpolated positions."""
    model.train(); model.rope_scale = scale
    d = data.train
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr, warmup=20)
        ix = torch.randint(len(d) - L - 1, (batch,))
        x = torch.stack([d[i:i + L] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + L] for i in ix])
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"    finetune step {step}/{steps} | loss {loss.item():.3f} | "
                  f"{time.time()-t0:.0f}s", flush=True)
    return model


def plot(pos, naive, pi, train_len, path):
    import plot_style as ps
    fig, ax = ps.new_axes(7.4, 4.4)
    ax.plot(pos, naive, color=ps.SERIES[2], label="naive RoPE (no rescale)", linewidth=1.3, alpha=0.9)
    ax.plot(pos, pi, color=ps.SERIES[0], label="interpolation + short finetune", linewidth=1.3, alpha=0.9)
    ax.axvline(train_len, color=ps.BASELINE, linestyle="--", linewidth=1.2)
    ax.text(train_len + 4, ax.get_ylim()[1] * 0.96, "trained window", fontsize=8,
            color=ps.INK_MUTED, va="top")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Predicting past the training window: naive degrades, interpolation holds",
              "position in sequence", "next-token loss (nats/char)", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "data/corpus.txt"))
    ap.add_argument("--eval-len", type=int, default=384)
    ap.add_argument("--n-seqs", type=int, default=40)
    ap.add_argument("--finetune-steps", type=int, default=250)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); torch.set_num_threads(12)

    ckpt = torch.load(CKPT, weights_only=False)
    cfg = ckpt["cfg"]
    text = Path(args.corpus).read_text(encoding="utf-8")
    data = CharData(text, block_size=cfg.block_size)
    train_len, L = cfg.block_size, args.eval_len
    scale = train_len / L

    # 1) naive extension: load the model, evaluate at L with no changes
    naive_model = GPT(cfg); naive_model.load_state_dict(ckpt["model"])
    print(f"loaded {naive_model.num_params()/1e6:.2f}M model trained at context {train_len}; "
          f"evaluating at {L}")
    naive = per_position_loss(naive_model, data, L, args.n_seqs, scale=1.0)

    # 2) position interpolation + a short fine-tune at the long context
    pi_model = GPT(cfg); pi_model.load_state_dict(ckpt["model"])
    print(f"position interpolation (scale {scale:.2f}) + fine-tuning {args.finetune_steps} steps ...")
    finetune_at_length(pi_model, data, L, scale, args.finetune_steps, batch=12)
    pi = per_position_loss(pi_model, data, L, args.n_seqs, scale=scale)

    pos = np.arange(L)
    plot(pos, naive, pi, train_len, OUT / "per_position_loss.png")

    beyond = slice(train_len, L)
    nb, pb = float(naive[beyond].mean()), float(pi[beyond].mean())
    (OUT / "results.csv").write_text(
        "metric,value\n"
        f"train_context,{train_len}\neval_context,{L}\n"
        f"loss_within_window_naive,{float(naive[:train_len].mean()):.3f}\n"
        f"loss_beyond_window_naive,{nb:.3f}\nperplexity_beyond_naive,{np.exp(nb):.1f}\n"
        f"loss_beyond_window_interpolated,{pb:.3f}\nperplexity_beyond_interpolated,{np.exp(pb):.1f}\n")
    print(f"beyond the {train_len}-token window: naive {nb:.3f} (ppl {np.exp(nb):.1f}) "
          f"vs interpolation+finetune {pb:.3f} (ppl {np.exp(pb):.1f})")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
