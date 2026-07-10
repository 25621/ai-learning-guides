"""Reward-hacking forensics: the score goes up, the answers get worse — find the cause.

We first reproduce reward hacking, then work backward through the three usual
suspects to find which one is really responsible:

  1. the reward model   (does it have a blind spot?)
  2. the KL penalty beta (is the policy allowed to run away from the reference?)
  3. the rollout         (is the policy even producing the behavior we score?)

Each suspect gets an intervention, and we watch which one closes the gap between the
reward-model score (up) and the true accuracy (down). The verdict: a blind reward
model is the root cause; a high beta is a tourniquet, not a cure.

    python forensics.py       # ~8 min on CPU

Reuses project 31 (reward model), project 32 (PPO loop), and the shared task.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "28-sft-a-1b-base-model"))
sys.path.insert(0, str(HERE.parent / "31-train-a-reward-model"))
sys.path.insert(0, str(HERE.parent / "32-ppo-rlhf-loop"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import sft_lib as L   # noqa: E402
import rm as RM       # noqa: E402
import ppo as PPO     # noqa: E402

OUT = HERE / "outputs"
PPO.ITERS = 30        # keep each run short for the drill


class VerifierReward:
    """An unhackable reward: decode the sequence and check the arithmetic. 1 or 0."""
    def reward(self, seqs):
        out = []
        for s in seqs:
            txt = L.decode(s)
            try:
                lhs, rhs = txt.split("=", 1)
                a, b = lhs.split("+")
                out.append(1.0 if L.is_correct(int(a), int(b), rhs) else 0.0)
            except Exception:
                out.append(0.0)
        return torch.tensor(out)


def train_rm(kind, steps=400, seed=0):
    torch.manual_seed(seed)
    rm = RM.RewardModel()
    opt = torch.optim.AdamW(rm.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.1)
    rng = random.Random(seed)
    for _ in range(steps):
        batch = RM.pref_pairs(64, rng, kind=kind)
        r_ch = rm.reward([c for c, _, _ in batch]); r_rej = rm.reward([r for _, r, _ in batch])
        loss = -F.logsigmoid(r_ch - r_rej).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return rm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    blind_rm = train_rm("random")                          # never saw a near-miss
    # A "better reward model" that could tell near-misses apart is what we'd want — but at
    # this scale the RM simply can't learn it (project 31). The reward you CAN trust is a
    # verifier: an exact checker with no blind spot to hack.
    print(f"blind RM near-miss acc {RM.pairwise_acc(blind_rm,'close'):.2f} "
          f"(the exploitable proxy)")

    sft_state = L.sft_model(steps=500).state_dict()
    conditions = [
        ("A: blind RM, no KL", blind_rm, 0.0),             # the bug
        ("B: blind RM, strong KL", blind_rm, 0.5),         # tourniquet (KL)
        ("C: verifier, no KL", VerifierReward(), 0.0),     # the cure (unhackable reward)
    ]
    rows = {}
    for name, rm, beta in conditions:
        h = PPO.ppo_run(sft_state, rm, beta)
        rows[name] = (h[0, 2], h[-1, 2])                   # true accuracy start, end
        print(f"[{name}] accuracy {h[0,2]:.3f} -> {h[-1,2]:.3f}", flush=True)

    plot(rows)
    with open(OUT / "results.csv", "w") as f:
        f.write("condition,accuracy_start,accuracy_end,accuracy_change\n")
        for name, (s, e) in rows.items():
            f.write(f"{name},{s:.3f},{e:.3f},{e - s:+.3f}\n")
    print(f"wrote {OUT/'forensics.png'} + results.csv")


def plot(rows):
    import plot_style as ps
    names = list(rows)
    deltas = [rows[n][1] - rows[n][0] for n in names]
    colors = [ps.SERIES[2] if d < -0.02 else ps.SERIES[1] for d in deltas]
    fig, ax = ps.new_axes(7.6, 4.4)
    bars = ax.bar(range(len(names)), deltas, color=colors, width=0.6)
    for i, d in enumerate(deltas):
        ax.text(i, d + (0.005 if d >= 0 else -0.02), f"{d:+.0%}", ha="center",
                fontsize=9, color=ps.INK_SECONDARY)
    ax.axhline(0, color=ps.BASELINE, lw=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["A: blind RM\nno KL\n(the bug)", "B: blind RM\nstrong KL\n(KL tourniquet)",
                        "C: verifier\nno KL\n(the cure)"], fontsize=8.5)
    ps.finish(ax.figure, ax, "Forensics: KL is a tourniquet; an unhackable reward is the cure",
              "", "change in true accuracy after PPO", OUT / "forensics.png")


if __name__ == "__main__":
    main()
