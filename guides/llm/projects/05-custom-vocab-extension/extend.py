"""Custom vocab extension: teach a model a new "alphabet" (chemical SMILES) by
adding tokens and growing its embedding matrix to match.

A tokenizer built for English shreds a SMILES string like "CC(=O)Oc1ccccc1C(=O)O"
(aspirin) into a long stream of single characters, because it never learned any
chemistry fragments. We add 256 SMILES tokens, resize the model's embedding table
so every new ID gets a vector, and verify two things:

  1. SMILES now tokenizes to far fewer tokens (the new vocab compresses it), and
  2. the model's English generation is byte-identical to before — resizing
     preserves the existing rows, so nothing it already knew is disturbed.

The new rows start UNTRAINED (random), so the model can't yet *use* chemistry —
extending the vocab is the plumbing; learning the new tokens' meaning is a later
training step.

    python extend.py      # ~2 min on CPU (downloads GPT-2, 124M)
"""

from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
MODEL = "gpt2"

# A small corpus of real drug SMILES to mine chemistry fragments from.
SMILES_CORPUS = [
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                 # caffeine
    "CC(=O)Oc1ccccc1C(=O)O",                         # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                    # ibuprofen
    "CN1CCC[C@H]1c1cccnc1",                          # nicotine
    "CC(=O)Nc1ccc(O)cc1",                            # paracetamol
    "OC(=O)c1ccccc1O",                               # salicylic acid
    "Clc1ccccc1",                                    # chlorobenzene
    "c1ccc2ccccc2c1",                                # naphthalene
    "C1CCCCC1", "c1ccccc1", "c1ccncc1", "c1cc[nH]c1",
    "CCO", "CCN", "CC(=O)O", "COc1ccccc1", "Nc1ccccc1",
    "O=C(O)c1ccccc1", "CC(=O)Nc1ccccc1", "FC(F)(F)c1ccccc1",
    "CC(C)(C)c1ccccc1", "OCC(O)CO", "C(C(=O)O)N", "CSCC(N)C(=O)O",
    "c1ccc(cc1)c1ccccc1", "O=S(=O)(N)c1ccccc1", "Brc1ccccc1",
    "CN(C)C=O", "CC#N", "C=CC=C", "C1=CC=CC=C1", "O=C=O",
]
TEST_SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "nicotine": "CN1CCC[C@H]1c1cccnc1",
}
PROMPTS = [
    "The capital of France is",
    "In the beginning, the universe was",
    "She opened the door and saw",
]


def build_smiles_tokens(existing_vocab, n=256):
    """Most frequent 2–6 char SMILES substrings not already in the vocabulary."""
    c = Counter()
    for s in SMILES_CORPUS:
        for L in range(2, 7):
            for i in range(len(s) - L + 1):
                c[s[i:i + L]] += 1
    # most frequent first, then fill to exactly n with rarer fragments
    out = []
    for frag, cnt in c.most_common():
        if frag not in existing_vocab:
            out.append(frag)
        if len(out) == n:
            break
    return out


@torch.no_grad()
def greedy(model, tok, prompt, k=18):
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=k, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def plot_compression(before, after, path):
    import sys
    sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
    import plot_style as ps
    import numpy as np
    names = list(TEST_SMILES)
    x = np.arange(len(names)); w = 0.38
    fig, ax = ps.new_axes(7.4, 4.2)
    ax.bar(x - w / 2, [before[n] for n in names], w, color=ps.SERIES[2], label="original GPT-2 vocab")
    ax.bar(x + w / 2, [after[n] for n in names], w, color=ps.SERIES[1], label="+256 SMILES tokens")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Tokens per molecule: before vs. after extending the vocab",
              "molecule (SMILES)", "tokens", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    torch.manual_seed(0)

    v0 = len(tok)
    emb0 = model.get_input_embeddings().weight.shape[0]
    params0 = model.get_input_embeddings().weight.numel()

    # SMILES token counts BEFORE, and baseline English generation
    smiles_before = {n: len(tok.encode(s)) for n, s in TEST_SMILES.items()}
    english_before = [greedy(model, tok, p) for p in PROMPTS]

    # --- extend the vocabulary ---
    new_tokens = build_smiles_tokens(set(tok.get_vocab()), n=256)
    n_added = tok.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tok))
    model.eval()

    v1 = len(tok)
    emb1 = model.get_input_embeddings().weight.shape[0]
    params1 = model.get_input_embeddings().weight.numel()

    # SMILES token counts AFTER, and English generation AFTER (must be identical)
    smiles_after = {n: len(tok.encode(s)) for n, s in TEST_SMILES.items()}
    english_after = [greedy(model, tok, p) for p in PROMPTS]
    english_identical = english_before == english_after

    plot_compression(smiles_before, smiles_after, OUT / "smiles_compression.png")

    # reports
    gen_lines = ["=== English generation is unchanged by the resize ===\n"]
    for p, b, a in zip(PROMPTS, english_before, english_after):
        gen_lines.append(f"prompt: {p!r}")
        gen_lines.append(f"  before: {b!r}")
        gen_lines.append(f"  after : {a!r}")
        gen_lines.append(f"  identical: {b == a}\n")
    (OUT / "generation.txt").write_text("\n".join(gen_lines))

    tot_b, tot_a = sum(smiles_before.values()), sum(smiles_after.values())
    (OUT / "results.csv").write_text(
        "metric,value\n"
        f"vocab_before,{v0}\nvocab_after,{v1}\ntokens_added,{n_added}\n"
        f"embedding_rows_before,{emb0}\nembedding_rows_after,{emb1}\n"
        f"embedding_params_before,{params0}\nembedding_params_after,{params1}\n"
        f"smiles_tokens_before,{tot_b}\nsmiles_tokens_after,{tot_a}\n"
        f"smiles_reduction,{1 - tot_a/tot_b:.3f}\n"
        f"english_generation_identical,{english_identical}\n")

    print(f"vocab {v0} -> {v1} (added {n_added})")
    print(f"embedding matrix {emb0}x{model.config.n_embd} -> {emb1}x{model.config.n_embd} "
          f"(+{params1-params0:,} params)")
    print(f"SMILES tokens (test set): {tot_b} -> {tot_a}  ({(1-tot_a/tot_b)*100:.0f}% fewer)")
    for n in TEST_SMILES:
        print(f"  {n:12s} {smiles_before[n]:>3} -> {smiles_after[n]:>3} tokens")
    print(f"English generation identical before/after resize: {english_identical}")
    print(f"  e.g. {english_after[0]!r}")
    print(f"wrote figures + {OUT/'results.csv'} + generation.txt")


if __name__ == "__main__":
    main()
