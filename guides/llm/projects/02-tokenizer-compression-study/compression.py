"""Tokenizer compression study: the same sentence costs very different token
counts across languages — and you pay per token.

We take Article 1 of the Universal Declaration of Human Rights (the same meaning,
professionally translated) in five languages and tokenize each with six real
production tokenizers. Then we chart tokens-per-byte and the "language tax" —
how many more tokens a non-English speaker pays for identical content.

    python compression.py      # ~1 min on CPU (tokenizer downloads cached)

Needs: tiktoken, transformers, sentencepiece. Tokenizers download from the HF hub
on first run and are cached.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
OUT = HERE / "outputs"

# UDHR Article 1 — identical meaning, five languages / four scripts.
TEXT = {
    "English": "All human beings are born free and equal in dignity and rights. "
               "They are endowed with reason and conscience and should act towards "
               "one another in a spirit of brotherhood.",
    "French": "Tous les êtres humains naissent libres et égaux en dignité et en "
              "droits. Ils sont doués de raison et de conscience et doivent agir "
              "les uns envers les autres dans un esprit de fraternité.",
    "Mandarin": "人人生而自由，在尊严和权利上一律平等。他们赋有理性和良心，"
                "并应以兄弟关系的精神相对待。",
    "Hindi": "सभी मनुष्य जन्म से स्वतंत्र तथा गरिमा और अधिकारों में समान होते हैं। "
             "उन्हें बुद्धि और अंतरात्मा प्राप्त है तथा उन्हें एक-दूसरे के प्रति भाईचारे की "
             "भावना से काम करना चाहिए।",
    "Bengali": "সমস্ত মানুষ স্বাধীনভাবে সমান মর্যাদা এবং অধিকার নিয়ে জন্মগ্রহণ করে। "
               "তাঁদের বিবেক এবং বুদ্ধি আছে; সুতরাং সকলেরই একে অপরের প্রতি ভ্রাতৃত্বসুলভ "
               "মনোভাব নিয়ে আচরণ করা উচিত।",
}
LANGS = list(TEXT)


def load_tokenizers():
    """Return {name: encode_fn(text)->list[int]} for six production tokenizers."""
    import tiktoken
    from transformers import AutoTokenizer

    toks = {}
    # OpenAI (tiktoken) — no special tokens added
    for name, enc in [("GPT-2", "gpt2"), ("GPT-4 (cl100k)", "cl100k_base"),
                      ("GPT-4o (o200k)", "o200k_base")]:
        e = tiktoken.get_encoding(enc)
        toks[name] = (lambda e: (lambda t: e.encode(t)))(e)
    # HuggingFace tokenizers — encode without special tokens for a fair count
    for name, repo in [("Llama 3", "NousResearch/Meta-Llama-3-8B"),
                       ("Gemma 2", "unsloth/gemma-2-9b"),
                       ("BLOOM", "bigscience/bloom-560m")]:
        tk = AutoTokenizer.from_pretrained(repo)
        toks[name] = (lambda tk: (lambda t: tk.encode(t, add_special_tokens=False)))(tk)
    return toks


def plot_heatmap(names, counts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    M = np.array([[counts[n][l] for l in LANGS] for n in names], dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=115)
    fig.patch.set_facecolor(ps.SURFACE)
    im = ax.imshow(M, cmap="magma_r", aspect="auto")
    ax.set_xticks(range(len(LANGS))); ax.set_xticklabels(LANGS, rotation=20, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(LANGS)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] > M.max() * 0.55 else "#0b0b0b", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("tokens", fontsize=9)
    ax.set_title("Tokens to encode the same sentence (UDHR Article 1)",
                 color=ps.INK, fontsize=12, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_tax(names, counts, path):
    """Language tax = tokens(lang) / tokens(English) per tokenizer, averaged."""
    import plot_style as ps
    fig, ax = ps.new_axes(7.6, 4.4)
    x = np.arange(len(LANGS))
    width = 0.13
    for i, n in enumerate(names):
        base = counts[n]["English"]
        mult = [counts[n][l] / base for l in LANGS]
        ax.bar(x + (i - len(names) / 2) * width, mult, width,
               color=ps.SERIES[i % len(ps.SERIES)], label=n)
    ax.axhline(1.0, color=ps.BASELINE, linewidth=1, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(LANGS, rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ps.finish(fig, ax, "The 'language tax': tokens relative to English (same meaning)",
              "language", "tokens vs. English (×)", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading tokenizers ...")
    toks = load_tokenizers()
    names = list(toks)

    counts, tpb = {}, {}
    for n, enc in toks.items():
        counts[n], tpb[n] = {}, {}
        for lang, text in TEXT.items():
            ids = enc(text)
            counts[n][lang] = len(ids)
            tpb[n][lang] = len(ids) / len(text.encode("utf-8"))

    plot_heatmap(names, counts, OUT / "token_heatmap.png")
    plot_tax(names, counts, OUT / "language_tax.png")

    # tables
    lines = ["tokenizer," + ",".join(LANGS)]
    for n in names:
        lines.append(n + "," + ",".join(str(counts[n][l]) for l in LANGS))
    (OUT / "token_counts.csv").write_text("\n".join(lines) + "\n")

    tlines = ["tokenizer," + ",".join(f"{l}_x_vs_english" for l in LANGS)]
    for n in names:
        base = counts[n]["English"]
        tlines.append(n + "," + ",".join(f"{counts[n][l]/base:.2f}" for l in LANGS))
    (OUT / "language_tax.csv").write_text("\n".join(tlines) + "\n")

    print("\ntokens to encode the same sentence:")
    print(f"  {'tokenizer':16s}" + "".join(f"{l[:8]:>9}" for l in LANGS))
    for n in names:
        print(f"  {n:16s}" + "".join(f"{counts[n][l]:>9}" for l in LANGS))
    print("\naverage 'tax' vs English across tokenizers:")
    for l in LANGS:
        avg = np.mean([counts[n][l] / counts[n]["English"] for n in names])
        print(f"  {l:10s} {avg:.2f}x")
    print(f"wrote figures + CSVs to {OUT}")


if __name__ == "__main__":
    main()
