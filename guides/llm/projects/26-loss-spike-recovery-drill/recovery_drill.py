"""Loss-spike recovery drill: catch a gradient spike, roll back, skip, resume —
and figure out how often you should be checkpointing in the first place.

Part 1 — the drill. Both runs are identical and *unclipped*, so they share a
trajectory until the spike. At step 300 an outlier (poisoned) batch sends the
gradient norm through the roof. The baseline applies it and is wrecked; the
guarded run trips a grad-norm threshold, rolls back to its last checkpoint, skips
the bad batch, and resumes — untouched.

Part 2 — the economics. Checkpoint too rarely and a spike costs you a lot of
recompute; checkpoint too often and the saving itself dominates. The Young-Daly
formula gives the sweet spot. We plot the trade-off and mark the optimum.

    python recovery_drill.py       # ~3 min on CPU

Reuses the GPT skeleton from project 08.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT, CharData, cosine_lr, estimate_loss   # noqa: E402

OUT = HERE / "outputs"
STEPS = 550
BATCH = 32
SPIKE_STEP = 300
SPIKE_LEN = 8                # a burst of outlier batches (one is barely a scratch)
CKPT_EVERY = 50
GRADNORM_THRESH = 3.0        # healthy grad-norm ~0.7; a poisoned batch spikes far past this
DETECT_AFTER = 60


def poison(vocab):
    return (torch.randint(0, vocab, (BATCH, 128)), torch.randint(0, vocab, (BATCH, 128)))


def run_drill(mode, data):
    torch.manual_seed(0); torch.set_num_threads(12)
    cfg = Config(vocab_size=data.vocab_size, n_layer=4, n_head=4, n_embd=128, block_size=128)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    curve, gnorms, ckpt, interventions = [], [], None, 0
    step = 0
    while step <= STEPS:
        if step % 25 == 0 or step == STEPS:
            curve.append((step, estimate_loss(model, data, BATCH, iters=12)["val"]))
        if step % CKPT_EVERY == 0:                       # both runs checkpoint identically
            ckpt = (step, copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()))
        if step == STEPS:
            break
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, STEPS, 3e-3, warmup=60)
        poisoned = SPIKE_STEP <= step < SPIKE_STEP + SPIKE_LEN
        x, y = poison(data.vocab_size) if poisoned else data.batch("train", BATCH)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))   # measure, don't clip
        gnorms.append((step, gn))
        if mode == "guarded" and step >= DETECT_AFTER and gn > GRADNORM_THRESH and ckpt is not None:
            _, ms, os = ckpt
            model.load_state_dict(ms); opt.load_state_dict(os)   # roll back to last good weights
            interventions += 1
            step += 1                                            # skip this bad batch, try the next
            continue
        opt.step(); step += 1
    print(f"[{mode}] final val {curve[-1][1]:.3f} | interventions {interventions} "
          f"| peak grad-norm {max(g for _, g in gnorms):.1f}")
    return np.array(curve), interventions


def young_daly(t_step, ckpt_cost, mtbf_steps, total_steps=100_000):
    """Total wasted time vs checkpoint interval K, and the optimal K.

    wasted(K) = (checkpoints written) * ckpt_cost + (expected recompute on a failure)
              = (total/K)*ckpt_cost + (failures)*(K/2)*t_step
    Optimal K* = sqrt(2 * ckpt_cost * mtbf_steps / t_step)   (Young-Daly).
    """
    Ks = np.linspace(5, 400, 200)
    n_fail = total_steps / mtbf_steps
    ckpt_overhead = (total_steps / Ks) * ckpt_cost
    recompute = n_fail * (Ks / 2) * t_step
    total = ckpt_overhead + recompute
    kstar = np.sqrt(2 * ckpt_cost * mtbf_steps / t_step)
    return Ks, ckpt_overhead, recompute, total, kstar


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    text = Path(HERE / "data/corpus.txt").read_text(encoding="utf-8")
    data = CharData(text, block_size=128)

    base, _ = run_drill("baseline", data)
    guard, ninter = run_drill("guarded", data)

    # illustrative infra numbers: a 1s step, a 20s checkpoint, a spike every ~2000 steps
    Ks, over, recomp, total, kstar = young_daly(t_step=1.0, ckpt_cost=20.0, mtbf_steps=2000)

    plot(base, guard, ninter, Ks, over, recomp, total, kstar)
    with open(OUT / "results.csv", "w") as f:
        f.write("run,final_val_loss\n")
        f.write(f"baseline,{base[-1,1]:.3f}\nguarded,{guard[-1,1]:.3f}\n")
        f.write(f"guarded_interventions,{ninter}\n")
        f.write(f"young_daly_optimal_interval_steps,{kstar:.0f}\n")
    print(f"wrote {OUT/'recovery.png'}, {OUT/'checkpoint_cadence.png'} + results.csv")


def plot(base, guard, ninter, Ks, over, recomp, total, kstar):
    import plot_style as ps
    # 1) the drill
    fig, ax = ps.new_axes(7.2, 4.4)
    ax.plot(base[:, 0], np.clip(base[:, 1], 0, 5), color=ps.SERIES[2], label="baseline (no recovery)")
    ax.plot(guard[:, 0], np.clip(guard[:, 1], 0, 5), color=ps.SERIES[1],
            label=f"guarded (grad-norm rollback, {ninter} intervention)")
    ax.axvline(SPIKE_STEP, color=ps.BASELINE, ls="--", lw=1)
    ax.text(SPIKE_STEP + 6, ax.get_ylim()[1] * 0.9, "outlier batch", color=ps.INK_MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Grad-norm trip wire: rollback beats a wrecked run",
              "training step", "validation loss (nats/char)", OUT / "recovery.png")

    # 2) checkpoint cadence economics
    fig, ax = ps.new_axes(7.2, 4.4)
    ax.plot(Ks, over, color=ps.SERIES[3], label="checkpoint overhead (∝ 1/K)")
    ax.plot(Ks, recomp, color=ps.SERIES[0], label="recompute after a spike (∝ K)")
    ax.plot(Ks, total, color=ps.INK, lw=2, label="total wasted time")
    ax.axvline(kstar, color=ps.SERIES[2], ls="--", lw=1)
    ax.text(kstar + 6, ax.get_ylim()[1] * 0.8, f"Young-Daly\noptimum ≈ {kstar:.0f} steps",
            color=ps.INK_SECONDARY, fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ps.finish(fig, ax, "How often to checkpoint: the Young-Daly trade-off",
              "checkpoint interval K (steps)", "wasted time (arbitrary units)",
              OUT / "checkpoint_cadence.png")


if __name__ == "__main__":
    main()
