"""GRPO on a (toy) math task: sample a group of answers, keep learning from the
ones that check out — RL with a verifiable reward (RLVR), no reward model.

For each problem the policy samples G answers. The reward is simply whether the
answer is correct (a verifier, not a learned reward model). GRPO's trick: the
*advantage* of each answer is how much its reward beats the group's own average, so
no value network is needed. We then take a clipped, KL-regularized policy step
toward the answers that beat the group.

    python grpo.py       # ~6 min on CPU

Reuses the shared task (sft_lib) and the GPT skeleton from project 08.
"""

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "28-sft-a-1b-base-model"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import sft_lib as L   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
G = 8                 # completions sampled per prompt (the "group")
B = 32                # prompts per iteration
ITERS = 40
EPS = 0.2             # PPO clip
BETA = 0.08           # KL to the SFT reference
INNER = 2             # PPO epochs per rollout batch


def rollout(policy, rng):
    """Sample G completions for each of B prompts; build seqs, masks, rewards, advantages."""
    prompts = [L.sample_problem(rng)[:2] for _ in range(B)]
    flat = [p for p in prompts for _ in range(G)]                 # (B*G,) repeated
    comps = L.sample_batch(policy, flat, temperature=0.7)
    seqs, masks, rewards = [], [], []
    for (a, b), comp in zip(flat, comps):
        ids, mk = L.seq_from(a, b, comp)
        seqs.append(ids); masks.append(mk)
        rewards.append(1.0 if L.is_correct(a, b, comp) else 0.0)
    r = np.array(rewards).reshape(B, G)
    adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-4)
    return seqs, masks, torch.tensor(adv.reshape(-1), dtype=torch.float), r.mean()


def grpo_step(policy, ref, opt, seqs, masks, adv):
    old_lp, m = L.completion_token_logps(policy, seqs, masks)
    old_lp = old_lp.detach()
    ref_lp, _ = L.completion_token_logps(ref, seqs, masks); ref_lp = ref_lp.detach()
    for _ in range(INNER):
        new_lp, m = L.completion_token_logps(policy, seqs, masks)
        ratio = (new_lp - old_lp).exp()
        a = adv[:, None]
        pg = -torch.min(ratio * a, ratio.clamp(1 - EPS, 1 + EPS) * a)
        # k3 KL estimator to the frozen reference (per token, >= 0)
        d = ref_lp - new_lp
        kl = d.exp() - d - 1
        loss = ((pg + BETA * kl) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    return float(loss.item())


def main():
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    policy = L.sft_model(steps=600)                              # partial SFT baseline
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4, betas=(0.9, 0.95))
    rng = random.Random(0)

    start_acc = L.accuracy(policy)
    curve = [(0, start_acc, start_acc)]                          # (iter, acc, group_reward)
    print(f"SFT baseline accuracy {start_acc:.3f}")
    for it in range(1, ITERS + 1):
        seqs, masks, adv, mean_r = rollout(policy, rng)
        grpo_step(policy, ref, opt, seqs, masks, adv)
        if it % 5 == 0 or it == ITERS:
            acc = L.accuracy(policy)
            curve.append((it, acc, mean_r))
            print(f"  iter {it}/{ITERS} | train group-reward {mean_r:.2f} | acc {acc:.3f}",
                  flush=True)
    final_acc = curve[-1][1]
    torch.save({"model": policy.state_dict()}, CKPT / "grpo.pt")
    plot(curve, start_acc)
    np.savez(CKPT / "curve.npz", curve=np.array(curve))
    (OUT / "results.csv").write_text(
        "stage,exact_answer_accuracy\n"
        f"sft_baseline,{start_acc:.3f}\ngrpo_final,{final_acc:.3f}\n")
    print(f"SFT {start_acc:.3f} -> GRPO {final_acc:.3f} | wrote {OUT/'grpo.png'} + results.csv")


def plot(curve, start_acc):
    import plot_style as ps
    c = np.array(curve)
    fig, ax = ps.new_axes(7.2, 4.4)
    ax.plot(c[:, 0], c[:, 1], "-o", color=ps.SERIES[1], label="exact-answer accuracy")
    ax.plot(c[:, 0], c[:, 2], "-o", color=ps.SERIES[3], label="train group-reward")
    ax.axhline(start_acc, color=ps.BASELINE, ls="--", lw=1)
    ax.text(1, start_acc + 0.02, "SFT start", color=ps.INK_MUTED, fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "GRPO improves a partial SFT model with only a verifier",
              "GRPO iteration", "accuracy / reward", OUT / "grpo.png")


if __name__ == "__main__":
    main()
