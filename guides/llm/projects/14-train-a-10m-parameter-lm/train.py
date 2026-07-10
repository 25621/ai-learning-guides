"""Train a ~10M-parameter character-level GPT on tiny-shakespeare.

This is the smallest *honest* pretraining run: the same forward -> loss ->
backward -> step loop that every frontier model uses, small enough to read every
line and watch next-token prediction work. Project 08 was a ~3M nanoGPT; here we
scale the identical skeleton to ~10M and focus on the loop itself.

    python train.py --corpus data/corpus.txt --steps 650      # ~10 min on CPU

Reuses the GPT skeleton from project 08.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT, CharData, cosine_lr, estimate_loss   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"


def plot_curve(curve, path):
    import plot_style as ps
    steps = [c[0] for c in curve]
    fig, ax = ps.new_axes(7.2, 4.2)
    ax.plot(steps, [c[1] for c in curve], color=ps.SERIES[0], label="train loss")
    ax.plot(steps, [c[2] for c in curve], color=ps.SERIES[2], label="val loss")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "A 10M-parameter LM learning Shakespeare",
              "training step", "cross-entropy loss (nats/char)", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "data/corpus.txt"))
    ap.add_argument("--steps", type=int, default=650)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); torch.set_num_threads(12)

    text = Path(args.corpus).read_text(encoding="utf-8")
    data = CharData(text, block_size=128)
    cfg = Config(vocab_size=data.vocab_size, n_layer=6, n_head=6, n_embd=384, block_size=128)
    model = GPT(cfg)
    print(f"model: {model.num_params()/1e6:.2f}M params | vocab {data.vocab_size} | "
          f"{cfg.n_layer} layers x d{cfg.n_embd} | RoPE + RMSNorm + SwiGLU")

    # ---- the entire pretraining loop, spelled out ----
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    curve, t0 = [], time.time()
    for step in range(args.steps + 1):
        for g in opt.param_groups:                       # warmup then cosine decay
            g["lr"] = cosine_lr(step, args.steps, 3e-3, warmup=100)
        if step % 50 == 0 or step == args.steps:
            losses = estimate_loss(model, data, args.batch)
            curve.append((step, losses["train"], losses["val"]))
            print(f"  step {step}/{args.steps} | train {losses['train']:.3f} | "
                  f"val {losses['val']:.3f} | {step/(time.time()-t0+1e-9):.2f} it/s", flush=True)
        if step == args.steps:
            break
        x, y = data.batch("train", args.batch)            # (B, T) inputs, (B, T) shifted targets
        logits, loss = model(x, y)                        # teacher forcing: true prev tokens
        opt.zero_grad(); loss.backward()                  # the six-line objective is inside model()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    torch.save({"model": model.state_dict(), "cfg": cfg, "stoi": data.stoi}, CKPT / "gpt10m.pt")
    plot_curve(curve, OUT / "loss_curve.png")

    torch.manual_seed(1)
    gen = model.generate(data.encode("\n"), max_new_tokens=600, temperature=0.8)
    sample = data.decode(gen[0].tolist())
    (OUT / "sample.txt").write_text(sample)
    print("\n=== sample ===\n" + sample[:700])

    (OUT / "results.csv").write_text(
        "metric,value\n"
        f"params_millions,{model.num_params()/1e6:.2f}\nvocab_size,{data.vocab_size}\n"
        f"final_train_loss,{curve[-1][1]:.3f}\nfinal_val_loss,{curve[-1][2]:.3f}\n"
        f"steps,{args.steps}\n")
    print(f"\nwrote {OUT/'loss_curve.png'} + sample.txt + results.csv")


if __name__ == "__main__":
    main()
