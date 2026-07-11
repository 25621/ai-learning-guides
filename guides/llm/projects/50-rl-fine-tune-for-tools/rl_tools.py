"""Project 50 — RL fine-tune for tools: GRPO against a tool-call verifier.

Start from a tool user SFT'd on a *corrupted* trace corpus: 55% of its
compose traces skip the tools and blurt a guessed answer, so the served
(greedy) behavior on compose questions is a hallucinated number, while the
correct lookup->lookup->calc chain survives only as a minority mode the
sampler occasionally visits. Then run GRPO where the reward is pure RLVR:
the orchestrator executes the episode's tool calls, and a verifier pays 1.0
iff the final `A:` answer equals the ground truth. No reward model, nothing
to hack — the tools themselves are the judge. The reward finds the latent
chain mode, and the policy flips to it within a handful of iterations.

The multi-turn twist over Phase 5's GRPO: rollouts are *interactive* (the
environment splices `R:...;` observations into the sequence mid-generation),
so log-probs, the PPO ratio, and the KL penalty are all computed under the
model-token mask from `tool_lib.run_episodes` — environment tokens carry no
policy gradient. Everything else reuses the Phase-5/6 recipe: temp-0.7
rollouts, group-relative advantages, token-level clipping, k3 KL to the
frozen SFT reference — except the learning rate; see LR below.

The ablation CSVs in outputs/ are earlier full runs from *uncorrupted*
partial-SFT starts (polished 600-step start at lr 3e-4; brittle 560-step
cliff start at 3e-4 and 1e-4). None of them improve on their baseline:
their residual errors are unstructured copy-slips, which a 1-bit episode
reward cannot teach away.
"""

import csv
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "47-tool-using-chatbot"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))

import plot_style as ps  # noqa: E402
import tool_lib as tl  # noqa: E402

OUT = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoints")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

SFT_STEPS = 1000      # full training, but on a *corrupted* trace mix
LAZY_FRAC = 0.55      # this share of compose traces skip the tools and
                      # blurt a guessed answer — a systematic bad habit for
                      # RL to unlearn (see the ablations for why RL needs a
                      # structured fault, not random noise, to fix). At 55%
                      # the lazy branch wins the greedy argmax: the served
                      # behavior is the bug, the tool chain a latent mode.
ITERS = 27
N_PROMPTS = 16        # questions per GRPO iteration
G = 8                 # rollouts per question
TEMP = 0.7
LR = 1.5e-4           # gentler than Phase 5's 3e-4: here one advantage is
                      # spread over 10-25 model tokens per episode, so the
                      # effective policy step is much larger per iteration
BETA = 0.08           # k3-KL weight to the frozen SFT reference
CLIP = 0.2
EVAL_EVERY = 3
N_EVAL = 150


def lazy_traces(bs, rng):
    """Expert traces, except LAZY_FRAC of compose traces answer directly.

    The lazy branch imitates a real SFT-corpus disease: recorded traces
    where the annotator (or scraped agent) skipped the tools and guessed.
    The guess is format-plausible and *sometimes* right — exactly the kind
    of habit next-token imitation absorbs without complaint."""
    out = []
    for _ in range(bs):
        qd = tl.make_question(rng, rng.choice(tl.KINDS))
        if qd["kind"] == "compose" and rng.random() < LAZY_FRAC:
            guess = rng.randint(20, 198)
            pieces = [(qd["q"], 0), (f"A:{guess};", 1)]
        else:
            pieces = tl.expert_pieces(qd)
        out.append(tl.pieces_to_trace(pieces))
    return out


def weak_sft():
    path = os.path.join(CKPT, f"lazy_sft_{SFT_STEPS}.pt")
    model = tl.new_model(seed=0)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print("loaded cached lazy SFT")
    else:
        print(f"== SFT on corrupted traces ({SFT_STEPS} steps, "
              f"{LAZY_FRAC:.0%} lazy compose) ==", flush=True)
        tl.train_sft(model, lazy_traces, SFT_STEPS, tag="lazy ")
        torch.save(model.state_dict(), path)
    return model


def eval_all(model):
    accs = {}
    for kind in tl.KINDS:
        r = tl.qa_eval(model, kind, N_EVAL)
        accs[kind] = r["acc"]
        accs[f"{kind}_malformed"] = r["malformed"]
        if kind == "compose":
            # chain rate: episodes that actually worked the tools (2 lookups
            # + 1 calc) rather than blurting a direct answer
            accs["chain_rate"] = float(np.mean(
                [s["tool_calls"] >= 3 for s in r["states"]]))
    accs["mean"] = float(np.mean([accs[k] for k in tl.KINDS]))
    return accs


def rollout(model, rng):
    """N_PROMPTS questions x G interactive rollouts; verifier rewards."""
    qds = [tl.make_question(rng, rng.choice(tl.KINDS))
           for _ in range(N_PROMPTS)]
    prompts, states = [], []
    for qd in qds:
        for _ in range(G):
            prompts.append(qd["q"])
            states.append({"db": qd["db"], "ans": qd["ans"], "answer": None,
                           "malformed": 0, "tool_calls": 0})
    model.eval()
    seqs, masks = tl.run_episodes(model, prompts, tl.qa_step_fn, states,
                                  temperature=TEMP)
    rewards = torch.tensor([float(s["answer"] == str(s["ans"]))
                            for s in states])
    # group-relative advantage, clipped (Phase-5 lesson: std can be tiny)
    r = rewards.view(N_PROMPTS, G)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
    adv = adv.clamp(-3, 3).reshape(-1)
    return seqs, masks, rewards, adv


def grpo_step(model, ref, opt, seqs, masks, adv):
    with torch.no_grad():
        old_lp, mask = tl.episode_token_logps(model, seqs, masks)
        ref_lp, _ = tl.episode_token_logps(ref, seqs, masks)
    model.train()
    new_lp, _ = tl.episode_token_logps(model, seqs, masks)
    ratio = torch.exp(new_lp - old_lp)
    a = adv.unsqueeze(1)
    l_clip = -torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a)
    d = (ref_lp - new_lp).clamp(-8, 8)            # k3 KL, clamped
    kl = torch.exp(d) - d - 1
    loss = ((l_clip + BETA * kl) * mask).sum() / mask.sum().clamp(min=1.0)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss), float((kl * mask).sum() / mask.sum())


def main():
    model = weak_sft()
    ref = tl.new_model(seed=0)
    ref.load_state_dict(model.state_dict())
    ref.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))
    rng = random.Random(0)

    history = []
    accs = eval_all(model)
    history.append((0, accs))
    print(f"iter  0: " + "  ".join(f"{k} {accs[k]:.3f}" for k in tl.KINDS) +
          f"  mean {accs['mean']:.3f}  chain {accs['chain_rate']:.3f}",
          flush=True)

    for it in range(1, ITERS + 1):
        seqs, masks, rewards, adv = rollout(model, rng)
        loss, kl = grpo_step(model, ref, opt, seqs, masks, adv)
        line = (f"iter {it:2d}: reward {rewards.mean():.3f} "
                f"loss {loss:.3f} kl {kl:.4f}")
        if it % EVAL_EVERY == 0 or it == ITERS:
            accs = eval_all(model)
            history.append((it, accs))
            line += ("  |  " + "  ".join(f"{k} {accs[k]:.3f}"
                                         for k in tl.KINDS) +
                     f"  mean {accs['mean']:.3f}"
                     f"  chain {accs['chain_rate']:.3f}")
        print(line, flush=True)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter"] + list(tl.KINDS) + ["mean", "chain_rate"] +
                   [f"{k}_malformed" for k in tl.KINDS])
        for it, a in history:
            w.writerow([it] + [f"{a[k]:.3f}" for k in tl.KINDS] +
                       [f"{a['mean']:.3f}", f"{a['chain_rate']:.3f}"] +
                       [f"{a[f'{k}_malformed']:.3f}" for k in tl.KINDS])

    fig, ax = ps.new_axes(7.6, 4.4)
    its = [h[0] for h in history]
    for i, kind in enumerate(tl.KINDS):
        ax.plot(its, [h[1][kind] for h in history], "o-",
                color=ps.SERIES[i], linewidth=1.8, markersize=4, label=kind)
    ax.plot(its, [h[1]["chain_rate"] for h in history], "--",
            color=ps.SERIES[4], linewidth=1.8,
            label="compose: used the tool chain")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY,
              loc="lower right")
    ps.finish(fig, ax, "GRPO with a tool-output verifier (reward = final "
              "answer correct)", "GRPO iteration", "greedy accuracy",
              os.path.join(OUT, "rl_tools.png"))


if __name__ == "__main__":
    main()
