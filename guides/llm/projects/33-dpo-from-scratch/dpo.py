"""DPO from scratch: the result of RLHF without any reinforcement learning.

DPO replaces the reward model + PPO loop with a single supervised loss on
(chosen, rejected) pairs. The optimal RLHF policy has a closed form in terms of the
reference policy, and substituting it collapses the whole RL problem into:

    L = -log sigmoid( beta * [ (logp_pi(y_w) - logp_ref(y_w))
                              - (logp_pi(y_l) - logp_ref(y_l)) ] )

We (1) implement that loss, (2) verify it against an independent reference
implementation (the byte-for-byte version TRL uses, written out by hand — TRL
itself isn't installed here), and (3) train with it and watch accuracy rise as the
policy is pushed toward the chosen (correct) answers.

    python dpo.py       # ~4 min on CPU

Reuses the shared task (sft_lib) and the GPT skeleton from project 08.
"""

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "28-sft-a-1b-base-model"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import sft_lib as L   # noqa: E402

OUT = HERE / "outputs"
BETA = 0.1


def pref_batch(n, rng):
    chosen, rejected = [], []
    for _ in range(n):
        a, b, c = L.sample_problem(rng)
        w = rng.randint(0, 2 * L.MAXN)                       # a clearly-wrong alternative
        while w == c:
            w = rng.randint(0, 2 * L.MAXN)
        chosen.append(L.seq_from(a, b, f"{c};"))
        rejected.append(L.seq_from(a, b, f"{w};"))
    return chosen, rejected


def dpo_loss(policy, ref, chosen, rejected, beta=BETA):
    """Compact DPO loss using the shared sequence-logprob helper."""
    cs, cm = [c[0] for c in chosen], [c[1] for c in chosen]
    rs, rm = [r[0] for r in rejected], [r[1] for r in rejected]
    pi_w = L.completion_logprobs(policy, cs, cm)
    pi_l = L.completion_logprobs(policy, rs, rm)
    with torch.no_grad():
        ref_w = L.completion_logprobs(ref, cs, cm)
        ref_l = L.completion_logprobs(ref, rs, rm)
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    acc = (logits > 0).float().mean().item()                 # preference accuracy
    return -F.logsigmoid(logits).mean(), acc


def dpo_loss_reference(policy, ref, chosen, rejected, beta=BETA):
    """Independent reimplementation (the TRL formulation), for cross-checking.

    Builds padded tensors and gathers per-token log-probs with an explicit -100
    label mask, exactly as reference implementations do."""
    def seq_logp(model, pairs):
        T = max(len(s) for s, _ in pairs)
        x = torch.full((len(pairs), T), L.END, dtype=torch.long)
        labels = torch.full((len(pairs), T), -100)
        for i, (s, mk) in enumerate(pairs):
            x[i, :len(s)] = torch.tensor(s)
            for j, mj in enumerate(mk):
                if mj:
                    labels[i, j] = s[j]
        logits, _ = model(x[:, :-1])
        logp = F.log_softmax(logits, dim=-1)
        lab = labels[:, 1:]
        gathered = logp.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (gathered * (lab != -100)).sum(-1)

    pi_w, pi_l = seq_logp(policy, chosen), seq_logp(policy, rejected)
    with torch.no_grad():
        ref_w, ref_l = seq_logp(ref, chosen), seq_logp(ref, rejected)
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(logits).mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    ref = L.sft_model(steps=500)                             # frozen SFT reference
    policy = copy.deepcopy(ref)
    for p in ref.parameters():
        p.requires_grad_(False)

    # ---- (2) verify our loss == the independent reference implementation ----
    rng = random.Random(1)
    ch, rj = pref_batch(32, rng)
    ours, _ = dpo_loss(policy, ref, ch, rj)
    theirs = dpo_loss_reference(policy, ref, ch, rj)
    diff = abs(ours.item() - theirs.item())
    print(f"DPO loss check: ours {ours.item():.6f} | reference {theirs.item():.6f} | "
          f"diff {diff:.2e}")

    # ---- (3) train with DPO and track accuracy ----
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4, betas=(0.9, 0.95))
    start_acc = L.accuracy(policy)
    curve = [(0, start_acc)]
    for step in range(1, 401):
        ch, rj = pref_batch(64, rng)
        loss, pref_acc = dpo_loss(policy, ref, ch, rj)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 80 == 0:
            acc = L.accuracy(policy)
            curve.append((step, acc))
            print(f"  dpo step {step}/400 | loss {loss.item():.3f} | pref-acc "
                  f"{pref_acc:.2f} | exact-acc {acc:.3f}", flush=True)
    final_acc = curve[-1][1]
    plot(curve, start_acc)
    (OUT / "results.csv").write_text(
        "metric,value\n"
        f"loss_vs_reference_diff,{diff:.2e}\n"
        f"sft_baseline_accuracy,{start_acc:.3f}\ndpo_final_accuracy,{final_acc:.3f}\n")
    print(f"SFT {start_acc:.3f} -> DPO {final_acc:.3f} | wrote {OUT/'dpo.png'} + results.csv")


def plot(curve, start_acc):
    import plot_style as ps
    c = np.array(curve)
    fig, ax = ps.new_axes(7.0, 4.4)
    ax.plot(c[:, 0], c[:, 1], "-o", color=ps.SERIES[1])
    ax.axhline(start_acc, color=ps.BASELINE, ls="--", lw=1)
    ax.text(5, start_acc + 0.02, "SFT start", color=ps.INK_MUTED, fontsize=8)
    ax.set_ylim(0, 1)
    ps.finish(fig, ax, "DPO lifts accuracy from preference pairs alone (no RL loop)",
              "DPO step", "exact-answer accuracy", OUT / "dpo.png")


if __name__ == "__main__":
    main()
