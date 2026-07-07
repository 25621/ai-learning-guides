"""MaskGIT-style masked-token image model: generate many tokens at once.

The autoregressive transformer (project 16) writes one token at a time — 49
sequential network calls per image. MaskGIT instead trains a *bidirectional*
transformer to fill in masked tokens, then samples by iterative parallel
decoding: start from an all-masked grid, predict every token at once, KEEP the
few it is most confident about, re-mask the rest, and repeat for a handful of
steps. A whole image in ~8 passes instead of 49.

    python maskgit.py --data-dir data      # ~6 min on CPU (tokenizer + model)

Reuses the MNIST tokenizer and transformer block from project 16 via sys.path.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "16-tiny-image-transformer"))
from mnist_tok import load_or_train_tokenizer, encode_all, K, SEQ, GRID  # noqa: E402
from transformer import Block  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
MASK = K            # extra token id for "masked"


class MaskGIT(nn.Module):
    def __init__(self, dim=128, heads=4, layers=3):
        super().__init__()
        self.tok = nn.Embedding(K + 1, dim)              # +1 for MASK
        self.pos = nn.Parameter(torch.zeros(1, SEQ, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, K)

    def forward(self, seq):                              # (B, SEQ) -> (B, SEQ, K)
        x = self.tok(seq) + self.pos
        for blk in self.blocks:                          # no causal mask: bidirectional
            x = blk(x)
        return self.head(self.ln(x))

    @torch.no_grad()
    def sample(self, n, steps=8, seed=0, temp=4.5):
        torch.manual_seed(seed)
        seq = torch.full((n, SEQ), MASK, dtype=torch.long)
        for t in range(steps):
            logits = self(seq)
            probs = F.softmax(logits, -1)
            sampled = torch.multinomial(probs.reshape(-1, K), 1).reshape(n, SEQ)
            conf = probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8).log()
            # MaskGIT's confidence noise: add annealed Gumbel noise so early steps
            # don't greedily commit to the single most-likely token everywhere
            # (that collapses every sample to the same "average" digit).
            noise = -torch.log(-torch.log(torch.rand_like(conf) + 1e-8) + 1e-8)
            conf = conf + temp * (1 - (t + 1) / steps) * noise
            known = seq != MASK
            sampled = torch.where(known, seq, sampled)                  # keep known tokens
            conf = torch.where(known, torch.full_like(conf, 1e9), conf)
            # how many tokens should remain masked after this step (cosine schedule)
            ratio = math.cos((t + 1) / steps * math.pi / 2)
            n_mask = 0 if t == steps - 1 else max(1, round(ratio * SEQ))
            n_keep = SEQ - n_mask
            # keep the n_keep most confident positions, re-mask the rest
            thresh = conf.sort(dim=1, descending=True).values[:, n_keep - 1:n_keep]
            keep = conf >= thresh
            seq = torch.where(keep, sampled, torch.full_like(seq, MASK))
        return seq


def gamma(u):
    """Cosine mask schedule: fraction of tokens to mask given u ~ U(0,1)."""
    return torch.cos(u * math.pi / 2)


def train(model, tokens, steps, bs=128, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(tokens), (bs,))
        seq = tokens[idx].clone()                        # (bs, SEQ)
        u = torch.rand(bs, 1)
        n_mask = (gamma(u) * SEQ).clamp(min=1).long()    # per-example mask count
        rank = torch.rand(bs, SEQ).argsort(dim=1)        # random positions
        mask = rank < n_mask                             # (bs, SEQ) bool
        inp = torch.where(mask, torch.full_like(seq, MASK), seq)
        logits = model(inp)
        loss = F.cross_entropy(logits[mask], seq[mask])  # predict masked only
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 400 == 0:
            print(f"  [maskgit] step {step}/{steps} | loss {loss.item():.3f} | "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)


def plot_samples(imgs, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = imgs.size(0); cols = 10; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k < n:
            ax.imshow(imgs[k, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title, fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_speed(path):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps
    fig, ax = ps.new_axes(6.0, 4.0)
    names = ["autoregressive\n(project 16)", "MaskGIT\n(this)"]
    steps = [SEQ, 8]
    ax.bar(range(2), steps, color=[ps.SERIES[2], ps.SERIES[1]], width=0.6)
    for i, v in enumerate(steps):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=11, color=ps.INK_SECONDARY)
    ax.set_xticks(range(2)); ax.set_xticklabels(names)
    ps.finish(fig, ax, "Network passes to generate one image (fewer = faster)",
              "", "sequential decoding steps", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    print("training / loading MNIST tokenizer ...")
    tok = load_or_train_tokenizer(args.data_dir, CKPT / "tokenizer.pt")

    model = MaskGIT()
    if args.sample_only and (CKPT / "maskgit.pt").exists():
        model.load_state_dict(torch.load(CKPT / "maskgit.pt")["model"])
    else:
        print("encoding dataset to token sequences ...")
        tokens = encode_all(tok, args.data_dir, n=20000)
        print(f"MaskGIT params: {sum(p.numel() for p in model.parameters()):,}\ntraining ...")
        train(model, tokens, args.steps)
        CKPT.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, CKPT / "maskgit.pt")
    model.eval()

    print(f"sampling 30 images with {args.decode_steps} parallel decoding steps ...")
    t0 = time.time()
    seq = model.sample(30, steps=args.decode_steps)
    with torch.no_grad():
        imgs = tok.decode_indices(seq)
    secs = time.time() - t0
    plot_samples(imgs, OUT / "samples.png",
                 f"MaskGIT samples ({args.decode_steps} parallel steps, not 49)")
    plot_speed(OUT / "speed.png")
    (OUT / "results.csv").write_text(
        f"metric,value\nvocab,{K}\ntokens_per_image,{SEQ}\n"
        f"autoregressive_steps,{SEQ}\nmaskgit_steps,{args.decode_steps}\n"
        f"speedup,{SEQ/args.decode_steps:.1f}x\nsample_seconds_for_30,{secs:.2f}\n")
    print(f"generated 30 images in {secs:.1f}s using {args.decode_steps} steps "
          f"(vs {SEQ} for autoregressive — {SEQ/args.decode_steps:.1f}x fewer)")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
