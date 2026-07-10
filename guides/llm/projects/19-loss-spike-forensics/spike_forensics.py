"""Loss-spike forensics: blow the run up on purpose, then rehearse the recovery.

Real pretraining runs spike — an outlier batch or an optimizer wobble sends the
gradient norm through the roof and the loss with it. What separates a wasted week
of GPU time from a shrug is the detect -> rollback -> skip -> resume loop. Here we
inject a burst of poisoned (random-token) batches into an otherwise healthy run
and compare two responses:

  * naive   — no gradient clipping, no detection: the spike corrupts the weights
              and the loss stays elevated for the rest of the run.
  * guarded — clip the gradient, flag the grad-norm spike, roll back to the last
              checkpoint, skip the offending batches, and resume cleanly.

    python spike_forensics.py --run      # train both, record grad-norm + loss
    python spike_forensics.py --plot      # figures from saved results

Reuses the GPT skeleton from project 08.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT, CharData, cosine_lr, estimate_loss   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
STEPS = 550
BATCH = 32
SPIKE_STEP = 300
SPIKE_LEN = 12                # number of poisoned batches in the burst
CKPT_EVERY = 50
LOSS_THRESH = 3.0            # trip wire on the training loss (healthy loss is well under this)
DETECT_AFTER = 80            # arm detection only after warmup drops the loss below the threshold


def poison_batch(vocab, device="cpu"):
    """A batch of uniform-random tokens — its loss pins at log(vocab) no matter
    what the weights are, which is exactly why it is a reliable detection signal."""
    x = torch.randint(0, vocab, (BATCH, 128))
    y = torch.randint(0, vocab, (BATCH, 128))
    return x, y


def run_training(mode, data):
    torch.manual_seed(0); torch.set_num_threads(12)
    cfg = Config(vocab_size=data.vocab_size, n_layer=4, n_head=4, n_embd=128, block_size=128)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)

    curve, losshist = [], []
    ckpt = None                                          # (step, model_state, opt_state)
    interventions = []
    t0 = time.time()
    step = 0
    while step <= STEPS:
        if step % 25 == 0 or step == STEPS:
            losses = estimate_loss(model, data, BATCH)
            curve.append((step, losses["val"]))
        if mode == "guarded" and step % CKPT_EVERY == 0:
            ckpt = (step, copy.deepcopy(model.state_dict()),
                    copy.deepcopy(opt.state_dict()))
        if step == STEPS:
            break

        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, STEPS, 3e-3, warmup=60)

        poisoned = SPIKE_STEP <= step < SPIKE_STEP + SPIKE_LEN
        if poisoned:
            x, y = poison_batch(data.vocab_size)
        else:
            x, y = data.batch("train", BATCH)

        _, loss = model(x, y)
        lval = float(loss.item())
        losshist.append((step, lval))
        opt.zero_grad(); loss.backward()
        # naive: no clipping; guarded: clip to 1.0 (part of the recovery kit)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 if mode == "guarded" else 1e9)

        # Detection is loss-based: a poisoned batch pins the loss near log(vocab),
        # far above any healthy step, regardless of the current weights.
        if (mode == "guarded" and step >= DETECT_AFTER
                and lval > LOSS_THRESH and ckpt is not None):
            back_step, m_state, o_state = ckpt
            model.load_state_dict(m_state); opt.load_state_dict(o_state)   # roll back
            interventions.append((step, back_step))
            step = SPIKE_STEP + SPIKE_LEN               # skip past the poisoned batches
            continue

        opt.step()
        step += 1

    print(f"[{mode}] final val {curve[-1][1]:.3f} | interventions {len(interventions)} "
          f"| {STEPS/(time.time()-t0+1e-9):.2f} it/s")
    return np.array(curve), np.array(losshist), interventions


def run():
    CKPT.mkdir(parents=True, exist_ok=True)
    text = Path(HERE / "data/corpus.txt").read_text(encoding="utf-8")
    data = CharData(text, block_size=128)
    out = {}
    for mode in ("naive", "guarded"):
        curve, losshist, interv = run_training(mode, data)
        out[f"{mode}_curve"] = curve
        out[f"{mode}_loss"] = losshist
        out[f"{mode}_interv"] = np.array(interv) if interv else np.zeros((0, 2))
    np.savez(CKPT / "results.npz", **out)
    print("wrote checkpoints/results.npz")


def plot():
    import plot_style as ps
    OUT.mkdir(parents=True, exist_ok=True)
    r = np.load(CKPT / "results.npz")
    ninterv = int(r["guarded_interv"].shape[0])

    # 1) val loss: naive is scarred by the spike; guarded rolls back and recovers
    fig, ax = ps.new_axes(7.2, 4.2)
    nc, gc = r["naive_curve"], r["guarded_curve"]
    ax.plot(nc[:, 0], np.clip(nc[:, 1], 0, 5), color=ps.SERIES[2],
            label="naive (no clip, no rollback)")
    ax.plot(gc[:, 0], np.clip(gc[:, 1], 0, 5), color=ps.SERIES[1],
            label="guarded (detect + rollback + skip)")
    ax.axvspan(SPIKE_STEP, SPIKE_STEP + SPIKE_LEN, color=ps.BASELINE, alpha=0.4)
    ax.text(SPIKE_STEP + SPIKE_LEN + 6, ax.get_ylim()[1] * 0.9, "poisoned\nbatches",
            color=ps.INK_MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "A loss spike: scarred run vs. clean recovery",
              "training step", "validation loss (nats/char)", OUT / "loss_recovery.png")

    # 2) the per-step training loss — what the detector watches
    fig, ax = ps.new_axes(7.2, 3.8)
    nl = r["naive_loss"]
    ax.plot(nl[:, 0], np.clip(nl[:, 1], 0, 5), color=ps.SERIES[2], lw=0.9,
            label="naive training loss")
    ax.axhline(LOSS_THRESH, color=ps.SERIES[0], ls="--", lw=1)
    ax.text(6, LOSS_THRESH + 0.12, f"detection threshold = {LOSS_THRESH}",
            color=ps.INK_SECONDARY, fontsize=8)
    ax.axvspan(SPIKE_STEP, SPIKE_STEP + SPIKE_LEN, color=ps.BASELINE, alpha=0.4)
    ps.finish(fig, ax, "Training loss pins at log(vocab) on the poisoned batches",
              "training step", "per-step training loss", OUT / "detection.png")

    with open(OUT / "results.csv", "w") as f:
        f.write("run,final_val_loss,peak_train_loss,interventions\n")
        for m in ("naive", "guarded"):
            c = r[f"{m}_curve"]; ll = r[f"{m}_loss"]
            ni = int(r[f"{m}_interv"].shape[0])
            f.write(f"{m},{float(c[-1, 1]):.3f},{float(ll[:, 1].max()):.2f},{ni}\n")
    print(f"wrote {OUT/'loss_recovery.png'}, detection.png, results.csv "
          f"(guarded interventions: {ninterv})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.run:
        run()
    if args.plot:
        plot()


if __name__ == "__main__":
    main()
