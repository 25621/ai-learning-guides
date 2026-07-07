"""Tiny image transformer: generate images by writing their tokens like a sentence.

Once the MNIST tokenizer turns each digit into a 49-token sequence, a picture is
"just another sentence" and an autoregressive transformer generates it exactly
like a language model writes text — one token at a time, left to right, each token
conditioned on all the tokens before it. To make an image we sample 49 tokens in
sequence, reshape to 7x7, and decode back to pixels with the tokenizer.

    python imagegpt.py --data-dir data      # ~7 min on CPU (tokenizer + GPT)

Reuses the shared MNIST tokenizer (mnist_tok.py) and transformer block.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mnist_tok import load_or_train_tokenizer, encode_all, K, SEQ, GRID
from transformer import Block, causal_mask

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
BOS = K            # extra token id used to seed generation


class ImageGPT(nn.Module):
    def __init__(self, dim=128, heads=4, layers=3):
        super().__init__()
        self.tok = nn.Embedding(K + 1, dim)              # +1 for BOS
        self.pos = nn.Parameter(torch.zeros(1, SEQ, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, K)

    def forward(self, inp):                              # inp: (B, SEQ) = [BOS, t0..t47]
        x = self.tok(inp) + self.pos[:, :inp.size(1)]
        mask = causal_mask(inp.size(1), inp.device)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        return self.head(self.ln(x))                     # (B, SEQ, K)

    @torch.no_grad()
    def sample(self, n, temperature=1.0, seed=0):
        torch.manual_seed(seed)
        seq = torch.full((n, 1), BOS, dtype=torch.long)
        for _ in range(SEQ):
            logits = self(seq)[:, -1] / temperature
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            seq = torch.cat([seq, nxt], dim=1)
        return seq[:, 1:]                                # drop BOS -> (n, SEQ)


def train(model, tokens, steps, bs=128, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(tokens), (bs,))
        seq = tokens[idx]                                # (bs, SEQ) real tokens
        bos = torch.full((bs, 1), BOS, dtype=torch.long)
        inp = torch.cat([bos, seq[:, :-1]], dim=1)       # [BOS, t0..t47]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, K), seq.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 400 == 0:
            print(f"  [gpt] step {step}/{steps} | loss {loss.item():.3f} | "
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    print("training / loading MNIST tokenizer ...")
    tok = load_or_train_tokenizer(args.data_dir, CKPT / "tokenizer.pt")
    print("encoding dataset to token sequences ...")
    tokens = encode_all(tok, args.data_dir, n=20000)

    model = ImageGPT()
    print(f"ImageGPT params: {sum(p.numel() for p in model.parameters()):,}\ntraining ...")
    train(model, tokens, args.steps)
    model.eval()
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, CKPT / "gpt.pt")

    print("sampling 30 images autoregressively (49 sequential steps each) ...")
    t0 = time.time()
    seq = model.sample(30)
    with torch.no_grad():
        imgs = tok.decode_indices(seq)
    secs = time.time() - t0
    plot_samples(imgs, OUT / "samples.png",
                 "Autoregressive transformer samples (49 tokens generated left-to-right)")
    (OUT / "results.csv").write_text(
        f"metric,value\nvocab,{K}\ntokens_per_image,{SEQ}\n"
        f"sampling_steps_per_image,{SEQ}\nsample_seconds_for_30,{secs:.2f}\n")
    print(f"generated 30 images in {secs:.1f}s ({SEQ} steps each)")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
