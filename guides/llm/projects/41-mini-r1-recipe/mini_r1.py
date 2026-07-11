"""Mini R1 recipe: a tiny SFT warm-start, then RL on a correctness signal alone.

DeepSeek-R1's recipe in miniature: lightly SFT a model on reasoning traces
(chain-of-thought), then run GRPO with a *verifier* reward — "is the final answer
correct?" — and nothing else. No reward model, no human labels; the model improves
its reasoning purely from its own correct rollouts being reinforced.

    python mini_r1.py       # ~7 min on CPU

Reuses the shared task (reason_lib) and the GPT skeleton from project 08. The GRPO
step is the same one built in Phase 5's project 34, applied to multi-step reasoning.
"""

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "36-cot-vs-direct-on-gsm8k"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import reason_lib as R   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
G = 8; B = 32; ITERS = 40
EPS = 0.2; BETA = 0.08


def rollout(policy, rng):
    probs = [R.problem(rng) for _ in range(B)]
    flat = [xs for xs, _ in probs for _ in range(G)]
    comps = R.sample_batch(policy, flat, temperature=0.7)
    seqs, masks, rew = [], [], []
    for (xs, ans), c in zip([p for p in probs for _ in range(G)], comps):
        ids, mk = R.seq_from(xs, c); seqs.append(ids); masks.append(mk)
        rew.append(1.0 if R.is_correct(xs, ans, c) else 0.0)
    r = np.array(rew).reshape(B, G)
    adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-4)
    return seqs, masks, torch.tensor(adv.reshape(-1), dtype=torch.float), r.mean()


def grpo_step(policy, ref, opt, seqs, masks, adv):
    old_lp, m = R.completion_token_logps(policy, seqs, masks); old_lp = old_lp.detach()
    ref_lp, _ = R.completion_token_logps(ref, seqs, masks); ref_lp = ref_lp.detach()
    for _ in range(2):
        new_lp, m = R.completion_token_logps(policy, seqs, masks)
        ratio = (new_lp - old_lp).exp(); a = adv[:, None]
        pg = -torch.min(ratio * a, ratio.clamp(1 - EPS, 1 + EPS) * a)
        d = (ref_lp - new_lp).clamp(-8, 8); kl = d.exp() - d - 1
        loss = ((pg + BETA * kl) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()


def main():
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    # small SFT warm-start on reasoning traces (deliberately light -> partial ability)
    policy = R.cot_model(steps=500)
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4, betas=(0.9, 0.95))
    rng = random.Random(0)

    start = R.accuracy(policy)
    curve = [(0, start, start)]
    print(f"SFT warm-start accuracy {start:.3f}")
    for it in range(1, ITERS + 1):
        seqs, masks, adv, mean_r = rollout(policy, rng)
        grpo_step(policy, ref, opt, seqs, masks, adv)
        if it % 5 == 0 or it == ITERS:
            acc = R.accuracy(policy)
            curve.append((it, acc, mean_r))
            print(f"  iter {it}/{ITERS} | group-reward {mean_r:.2f} | accuracy {acc:.3f}",
                  flush=True)
    final = curve[-1][1]
    torch.save({"model": policy.state_dict()}, CKPT / "r1.pt")
    plot(curve, start)
    (OUT / "results.csv").write_text(
        "stage,exact_answer_accuracy\n"
        f"sft_warmstart,{start:.3f}\nafter_grpo,{final:.3f}\n")
    print(f"SFT {start:.3f} -> R1/GRPO {final:.3f} | wrote {OUT/'mini_r1.png'} + results.csv")


def plot(curve, start):
    import plot_style as ps
    c = np.array(curve)
    fig, ax = ps.new_axes(7.2, 4.4)
    ax.plot(c[:, 0], c[:, 1], "-o", color=ps.SERIES[1], label="exact-answer accuracy")
    ax.plot(c[:, 0], c[:, 2], "-o", color=ps.SERIES[3], label="training group-reward")
    ax.axhline(start, color=ps.BASELINE, ls="--", lw=1)
    ax.text(1, start + 0.02, "SFT warm-start", color=ps.INK_MUTED, fontsize=8)
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "R1 in miniature: a verifier reward teaches reasoning",
              "GRPO iteration", "accuracy / reward", OUT / "mini_r1.png")


if __name__ == "__main__":
    main()
