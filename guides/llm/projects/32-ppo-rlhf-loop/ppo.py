"""PPO RLHF: chase the reward model's score, but stay tethered to where you started.

The full classic loop: an SFT policy, a reward model (project 31), and PPO updates
that maximize reward while a KL penalty (beta) holds the policy near the SFT
reference. The reward model here is deliberately *blind to near-misses* (trained
only against random-wrong answers), so it scores any plausible-looking number
highly. That sets up the money shot: turn the KL penalty off and the policy learns
to farm the reward model's score with confident wrong answers — reward up, true
accuracy down. That is reward hacking, and the KL term is the knob that prevents it.

    python ppo.py       # ~7 min on CPU

Reuses the shared task (sft_lib), the RewardModel (project 31), and project 08's GPT.
"""

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "28-sft-a-1b-base-model"))
sys.path.insert(0, str(HERE.parent / "31-train-a-reward-model"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import sft_lib as L   # noqa: E402
import rm as RM       # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
B = 64
ITERS = 40
EPS = 0.2
INNER = 2


def ppo_run(sft_state, reward_model, beta, seed=0):
    """One PPO run at a given KL penalty. Tracks RM reward and TRUE accuracy."""
    policy = L.new_model()
    policy.load_state_dict(sft_state)
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=1.5e-4, betas=(0.9, 0.95))
    rng = random.Random(seed)
    hist = [(0, reward_of(policy, reward_model, rng), L.accuracy(policy))]
    for it in range(1, ITERS + 1):
        prompts = [L.sample_problem(rng)[:2] for _ in range(B)]
        comps = L.sample_batch(policy, prompts, temperature=0.7)
        seqs, masks = [], []
        for (a, b), c in zip(prompts, comps):
            ids, mk = L.seq_from(a, b, c); seqs.append(ids); masks.append(mk)
        with torch.no_grad():
            reward = reward_model.reward(seqs)
        adv = ((reward - reward.mean()) / (reward.std() + 1e-4)).clamp(-3, 3)
        old_lp, m = L.completion_token_logps(policy, seqs, masks); old_lp = old_lp.detach()
        ref_lp, _ = L.completion_token_logps(ref, seqs, masks); ref_lp = ref_lp.detach()
        for _ in range(INNER):
            new_lp, m = L.completion_token_logps(policy, seqs, masks)
            ratio = (new_lp - old_lp).exp()
            a = adv[:, None]
            pg = -torch.min(ratio * a, ratio.clamp(1 - EPS, 1 + EPS) * a)
            d = (ref_lp - new_lp).clamp(-8, 8)
            kl = d.exp() - d - 1
            loss = ((pg + beta * kl) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        if it % 5 == 0 or it == ITERS:
            hist.append((it, reward_of(policy, reward_model, rng), L.accuracy(policy)))
    return np.array(hist)


@torch.no_grad()
def reward_of(policy, reward_model, rng, n=128):
    prompts = [L.sample_problem(rng)[:2] for _ in range(n)]
    comps = L.sample_batch(policy, prompts, temperature=1.0)
    seqs = [L.seq_from(a, b, c)[0] for (a, b), c in zip(prompts, comps)]
    return float(reward_model.reward(seqs).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    # a reward model with a blind spot: only ever saw random-wrong rejects
    rm = RM.RewardModel()
    opt = torch.optim.AdamW(rm.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.1)
    rng = random.Random(0)
    for step in range(400):
        batch = RM.pref_pairs(64, rng, kind="random")
        r_ch = rm.reward([c for c, _, _ in batch]); r_rej = rm.reward([r for _, r, _ in batch])
        loss = -torch.nn.functional.logsigmoid(r_ch - r_rej).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"reward model trained | near-miss pairwise acc {RM.pairwise_acc(rm, 'close'):.2f} "
          f"(blind spot), random {RM.pairwise_acc(rm, 'random'):.2f}")

    sft_state = L.sft_model(steps=500).state_dict()

    runs = {}
    for name, beta in [("strong KL (beta=0.5)", 0.5), ("no KL (beta=0)", 0.0)]:
        runs[name] = ppo_run(sft_state, rm, beta)
        h = runs[name]
        print(f"[{name}] reward {h[0,1]:.2f}->{h[-1,1]:.2f} | accuracy "
              f"{h[0,2]:.3f}->{h[-1,2]:.3f}", flush=True)

    plot(runs)
    with open(OUT / "results.csv", "w") as f:
        f.write("run,reward_start,reward_end,accuracy_start,accuracy_end\n")
        for name, h in runs.items():
            f.write(f"{name},{h[0,1]:.2f},{h[-1,1]:.2f},{h[0,2]:.3f},{h[-1,2]:.3f}\n")
    print(f"wrote {OUT/'ppo.png'} + results.csv")


def plot(runs):
    import plot_style as ps
    import matplotlib.pyplot as plt
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
    colors = {"strong KL (beta=0.5)": ps.SERIES[1], "no KL (beta=0)": ps.SERIES[2]}
    for name, h in runs.items():
        ax1.plot(h[:, 0], h[:, 1], "-o", color=colors[name], label=name)
        ax2.plot(h[:, 0], h[:, 2], "-o", color=colors[name], label=name)
    ax1.set_title("Reward-model score (what PPO maximizes)", color=ps.INK, fontsize=10.5,
                  loc="left", pad=10)
    ax1.set_xlabel("PPO iteration", color=ps.INK_SECONDARY, fontsize=10)
    ax1.set_ylabel("mean reward", color=ps.INK_SECONDARY, fontsize=10)
    ax1.legend(frameon=False, fontsize=8)
    ax2.set_title("True accuracy (what we actually want)", color=ps.INK, fontsize=10.5,
                  loc="left", pad=10)
    ax2.set_xlabel("PPO iteration", color=ps.INK_SECONDARY, fontsize=10)
    ax2.set_ylabel("exact-answer accuracy", color=ps.INK_SECONDARY, fontsize=10)
    ax2.set_ylim(0, 1); ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ppo.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
