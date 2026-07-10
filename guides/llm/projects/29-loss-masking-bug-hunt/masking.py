"""Loss-masking bug hunt: grade the model on its answers, not on the questions.

SFT data is prompt + response, but only the response is the model's job. If you
compute the loss over *every* token (the one-line bug), the model spends capacity
learning to imitate the prompts — here, the random "a+b=" questions it can never
predict — instead of learning to answer. We run SFT twice, identical except for the
loss mask, and compare.

    python masking.py       # ~3 min on CPU

Reuses the shared task (sft_lib) and the GPT skeleton from project 08.
"""

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
STEPS = 650


def train_track(full_loss):
    m = L.new_model(seed=0)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    rng = random.Random(0)
    curve = []
    for step in range(STEPS + 1):
        if step % 50 == 0 or step == STEPS:
            curve.append((step, L.accuracy(m, n=200)))
        if step == STEPS:
            break
        x, y, mask = L.sft_batch(64, rng, full_loss=full_loss)
        logits, _ = m(x)
        loss = L.masked_ce(logits, y, mask)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m, np.array(curve)


def free_generation(model, n=200):
    """Prompt the model with nothing and see how much 'question' text it emits —
    the fingerprint of an unmasked fine-tune that learned to imitate prompts."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    starts = torch.zeros(n, 1, dtype=torch.long)          # a lone '0'
    ids = starts
    for _ in range(6):
        logits, _ = model(ids[:, -L.BLOCK:])
        nxt = torch.multinomial(F.softmax(logits[:, -1], -1), 1)
        ids = torch.cat([ids, nxt], 1)
    txt = [L.decode(r.tolist()) for r in ids]
    frac_plus = sum("+" in t for t in txt) / n            # emitted a new "a+b" question?
    return frac_plus


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    masked, c_masked = train_track(full_loss=False)
    full, c_full = train_track(full_loss=True)
    acc_masked, acc_full = c_masked[-1, 1], c_full[-1, 1]
    plus_masked = free_generation(masked)
    plus_full = free_generation(full)
    print(f"assistant-only mask: accuracy {acc_masked:.3f} | free-gen '+' rate {plus_masked:.2f}")
    print(f"full-sequence loss : accuracy {acc_full:.3f} | free-gen '+' rate {plus_full:.2f}")

    plot(c_masked, c_full, plus_masked, plus_full)
    (OUT / "results.csv").write_text(
        "loss,final_accuracy,free_gen_plus_rate\n"
        f"assistant_only_mask,{acc_masked:.3f},{plus_masked:.2f}\n"
        f"full_sequence,{acc_full:.3f},{plus_full:.2f}\n")
    print(f"wrote {OUT/'masking.png'} + results.csv")


def plot(c_masked, c_full, plus_masked, plus_full):
    import plot_style as ps
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4), dpi=110,
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor(ps.SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(ps.BASELINE)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=9)
        ax.grid(True, color=ps.GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.patch.set_facecolor(ps.SURFACE)
    ax1.plot(c_masked[:, 0], c_masked[:, 1], color=ps.SERIES[1], label="assistant-only mask")
    ax1.plot(c_full[:, 0], c_full[:, 1], color=ps.SERIES[2], label="full-sequence loss")
    ax1.set_ylim(0, 1); ax1.legend(frameon=False, fontsize=9)
    ax1.set_title("Masking the prompt trains faster and lands higher", color=ps.INK,
                  fontsize=10.5, loc="left", pad=10)
    ax1.set_xlabel("SFT step", color=ps.INK_SECONDARY, fontsize=10)
    ax1.set_ylabel("exact-answer accuracy", color=ps.INK_SECONDARY, fontsize=10)
    bars = ax2.bar(["mask", "full"], [plus_masked, plus_full],
                   color=[ps.SERIES[1], ps.SERIES[2]], width=0.55)
    for b, v in zip(bars, [plus_masked, plus_full]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center",
                 fontsize=9, color=ps.INK_SECONDARY)
    ax2.set_ylim(0, 1)
    ax2.set_title("Unmasked learns to emit questions", color=ps.INK, fontsize=10.5,
                  loc="left", pad=10)
    ax2.set_ylabel("free-gen rate of a new 'a+b' prompt", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "masking.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
