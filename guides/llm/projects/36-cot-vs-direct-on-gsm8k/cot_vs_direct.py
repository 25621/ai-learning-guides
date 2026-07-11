"""Chain-of-thought vs. direct answering: show your work and get it right more often.

We give a small model a four-number sum and train it two ways: to answer directly
("21;") or to think in tokens first ("3+7=10,10+9=19,19+2=21;"). The direct model
must compute a 4-way sum in one shot; the chain-of-thought model decomposes it into
three easy 2-number additions. Same architecture, same data, same steps — only the
output format differs.

    python cot_vs_direct.py       # ~5 min on CPU

Defines the shared task in reason_lib.py; reuses the GPT skeleton from project 08.
"""

import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import reason_lib as R   # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
STEPS = 800


def avg_tokens(model, n=100):
    rng = random.Random(7)
    probs = [R.problem(rng) for _ in range(n)]
    comps = R.sample_batch(model, [xs for xs, _ in probs], temperature=0.0)
    return sum(len(c) for c in comps) / n


def main():
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    direct = R.new_model(); R.train_sft(direct, STEPS, mode="direct")
    cot = R.new_model(); R.train_sft(cot, STEPS, mode="cot")
    torch.save({"model": cot.state_dict()}, CKPT / "cot.pt")

    acc_direct, acc_cot = R.accuracy(direct), R.accuracy(cot)
    tok_direct, tok_cot = avg_tokens(direct), avg_tokens(cot)
    print(f"direct: acc {acc_direct:.3f} | {tok_direct:.1f} tokens/answer")
    print(f"cot   : acc {acc_cot:.3f} | {tok_cot:.1f} tokens/answer")

    # sample the same problems
    rng = random.Random(3)
    probs = [R.problem(rng) for _ in range(6)]
    dc = R.sample_batch(direct, [xs for xs, _ in probs], temperature=0.0)
    cc = R.sample_batch(cot, [xs for xs, _ in probs], temperature=0.0)
    lines = []
    for (xs, ans), d, c in zip(probs, dc, cc):
        ok = "OK" if R.is_correct(xs, ans, c) else "X"
        lines.append(f"{R.prompt_str(xs)}{ans}   direct-> {d:<6}  cot-> {c} [{ok}]")
    sample_text = "\n".join(lines)
    (OUT / "samples.txt").write_text(sample_text)
    print("\n" + sample_text)

    plot(acc_direct, acc_cot, tok_direct, tok_cot)
    (OUT / "results.csv").write_text(
        "method,accuracy,tokens_per_answer\n"
        f"direct,{acc_direct:.3f},{tok_direct:.1f}\n"
        f"chain_of_thought,{acc_cot:.3f},{tok_cot:.1f}\n")
    print(f"\nwrote {OUT/'cot_vs_direct.png'} + samples.txt + results.csv")


def plot(acc_direct, acc_cot, tok_direct, tok_cot):
    import plot_style as ps
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.4), dpi=110)
    for ax in (ax1, ax2):
        ax.set_facecolor(ps.SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(ps.BASELINE)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=9)
        ax.grid(True, axis="y", color=ps.GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.patch.set_facecolor(ps.SURFACE)
    bars = ax1.bar(["direct", "chain-of-\nthought"], [acc_direct, acc_cot],
                   color=[ps.SERIES[2], ps.SERIES[1]], width=0.55)
    for b, v in zip(bars, [acc_direct, acc_cot]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center",
                 fontsize=10, color=ps.INK_SECONDARY)
    ax1.set_ylim(0, 1); ax1.set_title("Thinking in tokens wins", color=ps.INK, fontsize=11,
                                       loc="left", pad=10)
    ax1.set_ylabel("exact-answer accuracy", color=ps.INK_SECONDARY, fontsize=10)
    b2 = ax2.bar(["direct", "chain-of-\nthought"], [tok_direct, tok_cot],
                 color=[ps.SERIES[2], ps.SERIES[1]], width=0.55)
    for b, v in zip(b2, [tok_direct, tok_cot]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.0f}", ha="center",
                 fontsize=10, color=ps.INK_SECONDARY)
    ax2.set_title("...by spending more tokens", color=ps.INK, fontsize=11, loc="left", pad=10)
    ax2.set_ylabel("tokens generated per answer", color=ps.INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "cot_vs_direct.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
