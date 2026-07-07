"""Numeral tokenization audit: how tokenizers chop up numbers, and why it makes
arithmetic hard.

If "1234" is one token in one model and four in another, the two models see
numbers completely differently — and a model that never sees a number as a whole
unit has a much harder time doing math with it. We count tokens for every integer
0–9999 across several production tokenizers, chart how cost grows with digit
count, and show the exact digit-grouping each tokenizer uses.

    python numerals.py      # ~1 min on CPU (tokenizers cached from project 02)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
OUT = HERE / "outputs"


class Tok:
    """Uniform wrapper: .ntokens(s) and .pieces(s) (list of token strings)."""
    def __init__(self, encode, decode1):
        self._encode, self._decode1 = encode, decode1

    def ntokens(self, s):
        return len(self._encode(s))

    def pieces(self, s):
        return [self._decode1(i) for i in self._encode(s)]


def load_tokenizers():
    import tiktoken
    from transformers import AutoTokenizer
    toks = {}
    for name, enc in [("GPT-2", "gpt2"), ("GPT-4 (cl100k)", "cl100k_base"),
                      ("GPT-4o (o200k)", "o200k_base")]:
        e = tiktoken.get_encoding(enc)
        toks[name] = Tok((lambda e: (lambda s: e.encode(s)))(e),
                         (lambda e: (lambda i: e.decode([i])))(e))
    for name, repo in [("Llama 3", "NousResearch/Meta-Llama-3-8B"),
                       ("Gemma 2", "unsloth/gemma-2-9b")]:
        tk = AutoTokenizer.from_pretrained(repo)
        toks[name] = Tok((lambda tk: (lambda s: tk.encode(s, add_special_tokens=False)))(tk),
                         (lambda tk: (lambda i: tk.decode([i])))(tk))
    return toks


def plot_by_digits(names, per_digit, path):
    import plot_style as ps
    fig, ax = ps.new_axes(7.4, 4.4)
    digits = [1, 2, 3, 4]
    x = np.arange(len(digits)); w = 0.15
    for i, n in enumerate(names):
        ax.bar(x + (i - len(names) / 2) * w, [per_digit[n][d] for d in digits], w,
               color=ps.SERIES[i % len(ps.SERIES)], label=n)
    ax.set_xticks(x); ax.set_xticklabels([f"{d}-digit" for d in digits])
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ps.finish(fig, ax, "Average tokens per number, by digit count",
              "number of digits", "average tokens", path)


def plot_vs_value(names, tokens, path):
    import plot_style as ps
    fig, ax = ps.new_axes(7.6, 4.2)
    rng = range(100, 1300)
    for i, n in enumerate(names):
        ax.plot(list(rng), [tokens[n][v] for v in rng], color=ps.SERIES[i % len(ps.SERIES)],
                label=n, linewidth=1.2)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ps.finish(fig, ax, "Tokens per number across 100–1299 (watch the jumps at 1000)",
              "integer value", "tokens", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading tokenizers ...")
    toks = load_tokenizers()
    names = list(toks)

    N = 10000
    tokens = {n: [t.ntokens(str(v)) for v in range(N)] for n, t in toks.items()}

    # average tokens by digit count
    def digit_group(d):
        lo, hi = 10 ** (d - 1) if d > 1 else 0, 10 ** d
        return range(lo, hi)
    per_digit = {n: {d: float(np.mean([tokens[n][v] for v in digit_group(d)]))
                     for d in (1, 2, 3, 4)} for n in names}

    # consistency: how many DISTINCT token-counts appear among 3-digit numbers?
    consistency = {n: len(set(tokens[n][v] for v in range(100, 1000))) for n in names}

    plot_by_digits(names, per_digit, OUT / "tokens_by_digits.png")
    plot_vs_value(names, tokens, OUT / "tokens_vs_value.png")

    # exact digit grouping for a few example numbers
    examples = ["7", "42", "2023", "31415", "1000000"]
    lines = ["=== how each tokenizer splits example numbers ==="]
    for n in names:
        lines.append(f"\n{n}:")
        for ex in examples:
            pieces = toks[n].pieces(ex)
            lines.append(f"  {ex:>8}  ->  {len(pieces)} tokens  {pieces}")
    (OUT / "digit_grouping.txt").write_text("\n".join(lines) + "\n")

    clines = ["tokenizer,avg_1digit,avg_2digit,avg_3digit,avg_4digit,distinct_counts_3digit"]
    for n in names:
        d = per_digit[n]
        clines.append(f"{n},{d[1]:.2f},{d[2]:.2f},{d[3]:.2f},{d[4]:.2f},{consistency[n]}")
    (OUT / "results.csv").write_text("\n".join(clines) + "\n")

    print("\naverage tokens per number by digit count:")
    print(f"  {'tokenizer':16s}{'1d':>6}{'2d':>6}{'3d':>6}{'4d':>6}  distinct(3-digit)")
    for n in names:
        d = per_digit[n]
        print(f"  {n:16s}{d[1]:>6.2f}{d[2]:>6.2f}{d[3]:>6.2f}{d[4]:>6.2f}  {consistency[n]}")
    print("\nhow '31415' splits:")
    for n in names:
        print(f"  {n:16s} {toks[n].pieces('31415')}")
    print(f"wrote figures + {OUT/'results.csv'} + digit_grouping.txt")


if __name__ == "__main__":
    main()
