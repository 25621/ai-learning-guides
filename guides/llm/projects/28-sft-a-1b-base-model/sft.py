"""SFT: turn a base autocomplete into something that answers the question.

We first "pretrain" a base model on a noisy corpus — arithmetic in the right
*format* but with wrong answers (a stand-in for messy web text) — so it learns the
shape of the task without learning to solve it. Then supervised fine-tuning on
clean (prompt -> correct answer) demonstrations, with the loss masked to the
assistant's reply, teaches it to actually answer. We measure exact-answer accuracy
before and after.

    python sft.py       # ~4 min on CPU

Reuses the GPT skeleton from project 08; defines the shared toy task in sft_lib.py.
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import sft_lib as L   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"


def main():
    import random
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    probs = [(rng.randint(0, L.MAXN), rng.randint(0, L.MAXN)) for _ in range(6)]

    # 1) base "pretraining" on noisy, wrong-answer arithmetic (knows format, not math)
    model = L.new_model()
    L.train_sft(model, steps=150, corrupt=True, full_loss=True)
    base_acc = L.accuracy(model)
    base_samples = L.generate(model, probs)            # sample BEFORE SFT mutates it

    # 2) SFT on clean demonstrations, assistant-only loss mask (continue from base)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.1)
    L.train_sft(model, steps=750, corrupt=False, full_loss=False, opt=opt)
    sft_acc = L.accuracy(model)
    sft_samples = L.generate(model, probs)
    torch.save({"model": model.state_dict()}, CKPT / "sft.pt")
    print(f"base accuracy {base_acc:.3f} | SFT accuracy {sft_acc:.3f}")

    lines = []
    for (a, b), bg, sg in zip(probs, base_samples, sft_samples):
        ok = "OK" if L.is_correct(a, b, sg) else "X"
        lines.append(f"{a}+{b}={a+b}   base-> {bg:<5}   sft-> {sg:<5} [{ok}]")
    sample_text = "\n".join(lines)
    (OUT / "samples.txt").write_text(sample_text)
    print("\n" + sample_text)

    plot(base_acc, sft_acc)
    (OUT / "results.csv").write_text(
        "model,exact_answer_accuracy\n"
        f"base (noisy pretrain),{base_acc:.3f}\nsft (clean demos),{sft_acc:.3f}\n")
    print(f"\nwrote {OUT/'sft_accuracy.png'} + samples.txt + results.csv")


def plot(base_acc, sft_acc):
    import plot_style as ps
    fig, ax = ps.new_axes(6.2, 4.4)
    bars = ax.bar(["base\n(noisy pretrain)", "SFT\n(clean demos)"], [base_acc, sft_acc],
                  color=[ps.SERIES[2], ps.SERIES[1]], width=0.55)
    for b, v in zip(bars, [base_acc, sft_acc]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center",
                fontsize=10, color=ps.INK_SECONDARY)
    ax.set_ylim(0, 1)
    ps.finish(fig, ax, "SFT teaches the base to answer, not just autocomplete",
              "", "exact-answer accuracy", OUT / "sft_accuracy.png")


if __name__ == "__main__":
    main()
