"""Train a reward model: turn "this answer is better" into a number.

A reward model is a transformer with a scalar head instead of a vocabulary head:
it reads a prompt+answer and emits one number, the "reward". We train it on
preference pairs (for each problem, the correct answer is *chosen*, a wrong answer
is *rejected*) under the Bradley-Terry loss

    L = -log sigmoid( r(chosen) - r(rejected) )

and report how often it ranks the pair the way a human would. We also probe *where*
it's weakest: telling the right answer from a random wrong one is easy; telling it
from an off-by-one near-miss is much harder — the reward model's version of "humans
disagree on the close calls".

    python rm.py       # ~3 min on CPU

Reuses the shared task (sft_lib) and the GPT skeleton from project 08. The
RewardModel class here is imported by projects 32 (PPO) and 35 (reward hacking).
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "28-sft-a-1b-base-model"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import sft_lib as L   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"


class RewardModel(nn.Module):
    """A GPT body with a scalar reward head read off the last token."""
    def __init__(self, n_layer=4, n_head=4, n_embd=128):
        super().__init__()
        self.gpt = L.new_model(n_layer, n_head, n_embd)
        self.head = nn.Linear(n_embd, 1, bias=False)

    def hidden(self, x):
        g = self.gpt
        h = g.tok(x)
        rope = g._rope_for(x.size(1), x.device) if g.cfg.pos == "rope" else None
        for blk in g.blocks:
            h = blk(h, rope)
        return g.norm_f(h)

    def reward(self, seqs):
        """seqs: list[list[int]] of prompt+completion ids. Returns (B,) rewards
        read at each sequence's last real token."""
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), L.END, dtype=torch.long)
        last = []
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s); last.append(len(s) - 1)
        r = self.head(self.hidden(x)).squeeze(-1)          # (B, T)
        return r[torch.arange(len(seqs)), torch.tensor(last)]


def wrong_answer(a, b, rng, kind):
    """A rejected answer. 'random' is easy to reject; 'close' (off by 1-2) is hard."""
    c = a + b
    if kind == "close":
        w = c + rng.choice([-2, -1, 1, 2])
    else:
        w = rng.randint(0, 2 * L.MAXN)
        while w == c:
            w = rng.randint(0, 2 * L.MAXN)
    return max(0, w)


def pref_pairs(n, rng, kind="mixed"):
    pairs = []
    for _ in range(n):
        a, b, c = L.sample_problem(rng)
        k = kind if kind != "mixed" else rng.choice(["random", "close"])
        w = wrong_answer(a, b, rng, k)
        chosen, _ = L.seq_from(a, b, f"{c};")
        rejected, _ = L.seq_from(a, b, f"{w};")
        pairs.append((chosen, rejected, k))
    return pairs


def train_rm(steps=500, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    rm = RewardModel()
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = random.Random(seed)
    for step in range(steps):
        batch = pref_pairs(64, rng)
        r_ch = rm.reward([c for c, _, _ in batch])
        r_rej = rm.reward([r for _, r, _ in batch])
        loss = -F.logsigmoid(r_ch - r_rej).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"  rm step {step}/{steps} loss {loss.item():.3f}", flush=True)
    return rm


@torch.no_grad()
def pairwise_acc(rm, kind, n=600, seed=999):
    rng = random.Random(seed)
    pairs = pref_pairs(n, rng, kind=kind)
    r_ch = rm.reward([c for c, _, _ in pairs])
    r_rej = rm.reward([r for _, r, _ in pairs])
    return (r_ch > r_rej).float().mean().item()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)

    rm = train_rm(args.steps)
    torch.save({"model": rm.state_dict()}, CKPT / "rm.pt")
    acc_random = pairwise_acc(rm, "random")
    acc_close = pairwise_acc(rm, "close")
    acc_all = pairwise_acc(rm, "mixed")
    print(f"pairwise accuracy | random-wrong {acc_random:.3f} | close(off-by-1-2) "
          f"{acc_close:.3f} | mixed {acc_all:.3f}")

    plot(acc_random, acc_close, acc_all)
    (OUT / "results.csv").write_text(
        "reject_type,pairwise_accuracy\n"
        f"random_wrong,{acc_random:.3f}\nclose_off_by_1_2,{acc_close:.3f}\nmixed,{acc_all:.3f}\n")
    print(f"wrote {OUT/'rm_accuracy.png'} + results.csv")


def plot(a_rand, a_close, a_mixed):
    import plot_style as ps
    fig, ax = ps.new_axes(6.6, 4.4)
    bars = ax.bar(["random wrong\n(easy)", "off-by-1/2\n(hard)", "mixed"],
                  [a_rand, a_close, a_mixed],
                  color=[ps.SERIES[1], ps.SERIES[2], ps.SERIES[0]], width=0.6)
    for b, v in zip(bars, [a_rand, a_close, a_mixed]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center",
                fontsize=10, color=ps.INK_SECONDARY)
    ax.axhline(0.5, color=ps.BASELINE, ls="--", lw=1)
    ax.text(2.3, 0.52, "chance", color=ps.INK_MUTED, fontsize=8)
    ax.set_ylim(0, 1)
    ps.finish(fig, ax, "Reward model: easy to rank, hard on the close calls",
              "type of rejected answer", "pairwise accuracy", OUT / "rm_accuracy.png")


if __name__ == "__main__":
    main()
