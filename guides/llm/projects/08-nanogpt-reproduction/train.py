"""Train the nanoGPT skeleton on tiny-shakespeare and sample text.

A modern-flavored nanoGPT (RoPE + RMSNorm + SwiGLU, weight-tied head), trained at
the character level so it fits comfortably on a CPU. It learns Shakespeare's
spelling, spacing, and speaker-name format from scratch.

    python train.py --corpus data/corpus.txt --steps 3500      # ~8 min on CPU
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT, CharData, train_model   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"


def plot_curve(curve, path, title="nanoGPT training on tiny-shakespeare"):
    import plot_style as ps
    steps = [c[0] for c in curve]
    fig, ax = ps.new_axes(7.2, 4.2)
    ax.plot(steps, [c[1] for c in curve], color=ps.SERIES[0], label="train loss")
    ax.plot(steps, [c[2] for c in curve], color=ps.SERIES[2], label="val loss")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, title, "training step", "cross-entropy loss (nats/char)", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "data/corpus.txt"))
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); torch.set_num_threads(12)

    text = Path(args.corpus).read_text(encoding="utf-8")
    data = CharData(text, block_size=128)
    cfg = Config(vocab_size=data.vocab_size, n_layer=4, n_head=4, n_embd=128, block_size=128)
    model = GPT(cfg)
    print(f"model: {model.num_params()/1e6:.2f}M params | vocab {data.vocab_size} | "
          f"RoPE + RMSNorm + SwiGLU")

    curve = train_model(model, data, steps=args.steps, batch_size=args.batch, lr=3e-3,
                        warmup=100, eval_every=250)
    torch.save({"model": model.state_dict(), "cfg": cfg, "stoi": data.stoi}, CKPT / "gpt.pt")
    plot_curve(curve, OUT / "loss_curve.png")

    # sample text
    torch.manual_seed(1)
    start = data.encode("\n")
    gen = model.generate(start, max_new_tokens=500, temperature=0.8)
    sample = data.decode(gen[0].tolist())
    (OUT / "sample.txt").write_text(sample)
    print("\n=== sample ===\n" + sample[:600])

    (OUT / "results.csv").write_text(
        "metric,value\n"
        f"params_millions,{model.num_params()/1e6:.2f}\nvocab_size,{data.vocab_size}\n"
        f"final_train_loss,{curve[-1][1]:.3f}\nfinal_val_loss,{curve[-1][2]:.3f}\n"
        f"steps,{args.steps}\n")
    print(f"\nwrote figures + sample.txt + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
