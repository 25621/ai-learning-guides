"""Memorization probe: give the opening line, see if the model finishes verbatim.

Large language models reproduce a small but real fraction of their training data
word for word. We measure it directly: take texts that are near-certainly in
GPT-2's WebText training corpus (famous public-domain openings, anthems,
licenses) and feed the model only the first part. Then we greedily decode a
continuation and count how many tokens match the true text *exactly*.

Two controls make the result meaningful:

  * NOVEL text — original sentences written for this project, which the model
    cannot have seen. Verbatim reproduction here should be near zero: it isolates
    memorization from "the model is just good at English."
  * MODEL SIZE — we run the same probe on GPT-2 (124M) and GPT-2 Large (774M).
    Memorization grows with scale, so the larger model should regurgitate more.

We also break the result down text by text to see *which* famous openings the
model finishes word for word.

    python memorize.py          # ~6 min on CPU (two models, greedy decoding)
    python memorize.py --plot   # redraw from outputs/memorize.csv

CPU-only; models are queried, never trained.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
MODELS = ["gpt2", "gpt2-large"]


# --------------------------------------------------------------------------- #
# Canonical texts — famous, public-domain, and duplicated countless times
# across the web, so they are almost certainly memorized. Each is given whole;
# we split it into prefix + continuation at run time.
# --------------------------------------------------------------------------- #
CANONICAL = [
    "We hold these truths to be self-evident, that all men are created equal, "
    "that they are endowed by their Creator with certain unalienable Rights, "
    "that among these are Life, Liberty and the pursuit of Happiness.",

    "Four score and seven years ago our fathers brought forth on this "
    "continent, a new nation, conceived in Liberty, and dedicated to the "
    "proposition that all men are created equal.",

    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife.",

    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it was "
    "the epoch of incredulity.",

    "Call me Ishmael. Some years ago, never mind how long precisely, having "
    "little or no money in my purse, and nothing particular to interest me on "
    "shore, I thought I would sail about a little and see the watery part of "
    "the world.",

    "In the beginning God created the heaven and the earth. And the earth was "
    "without form, and void; and darkness was upon the face of the deep.",

    "Once upon a midnight dreary, while I pondered, weak and weary, over many a "
    "quaint and curious volume of forgotten lore.",

    "Twinkle, twinkle, little star, how I wonder what you are. Up above the "
    "world so high, like a diamond in the sky.",

    "Mary had a little lamb, its fleece was white as snow, and everywhere that "
    "Mary went the lamb was sure to go.",

    "To be, or not to be, that is the question: whether tis nobler in the mind "
    "to suffer the slings and arrows of outrageous fortune.",

    "Permission is hereby granted, free of charge, to any person obtaining a "
    "copy of this software and associated documentation files (the Software), "
    "to deal in the Software without restriction.",

    "The quick brown fox jumps over the lazy dog.",

    "Now is the winter of our discontent made glorious summer by this sun of "
    "York; and all the clouds that lowered upon our house in the deep bosom of "
    "the ocean buried.",

    "I have a dream that one day this nation will rise up and live out the true "
    "meaning of its creed: we hold these truths to be self-evident, that all "
    "men are created equal.",

    "Space: the final frontier. These are the voyages of the starship "
    "Enterprise. Its five-year mission: to explore strange new worlds.",

    "Happy families are all alike; every unhappy family is unhappy in its own "
    "way.",
]

# Novel text — written for this project, so not in any training set. Verbatim
# reproduction here should be near zero.
NOVEL = [
    "The copper kettle whistled twice before Marina remembered she had left "
    "the greenhouse door unlatched against the coming frost.",
    "According to the quarterly report, the Drennvale logistics hub rerouted "
    "sixteen shipments through the northern corridor last autumn.",
    "Professor Halloway argued that the migratory patterns of the estuary "
    "herons shifted noticeably after the third dredging season.",
    "My neighbor's grey tabby has developed the peculiar habit of napping only "
    "on the seventh stair, never the sixth or the eighth.",
    "The committee voted to postpone the annual regatta because the borrowed "
    "buoys had drifted past the western breakwater overnight.",
    "In the workshop behind the bakery, Tomas repaired antique barometers using "
    "a set of tools inherited from his grandmother.",
    "The spreadsheet flagged an anomaly in row forty-two, where the packaging "
    "costs exceeded the shipping estimates by a wide margin.",
    "She sketched the abandoned lighthouse from three angles before the fog "
    "rolled in and swallowed the rocky headland entirely.",
    "The recipe called for a pinch of smoked paprika, but Devlin substituted "
    "toasted fennel and declared the result a modest triumph.",
    "Our book club abandoned the mystery novel halfway through, unconvinced "
    "that the detective could have overlooked the muddy footprints.",
    "The city council installed twelve new benches along the riverwalk, each "
    "painted a slightly different shade of teal by local volunteers.",
    "A single loose thread on the tapestry led the conservators to a hidden "
    "signature stitched into the lower left corner of the frame.",
]


def longest_match_run(gen_ids, gold_ids):
    """Longest run of leading tokens that match exactly."""
    run = 0
    for g, t in zip(gen_ids, gold_ids):
        if g == t:
            run += 1
        else:
            break
    return run


@torch.no_grad()
def probe_model(name, prefix_frac=0.5):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(12)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()

    def run_set(texts):
        frac_matched, long_runs, best = [], [], None
        for text in texts:
            ids = tok(text, return_tensors="pt").input_ids[0]
            n = len(ids)
            cut = max(4, int(prefix_frac * n))
            prefix = ids[:cut].unsqueeze(0)
            gold = ids[cut:].tolist()
            if len(gold) < 4:
                continue
            gen = model.generate(prefix, max_new_tokens=len(gold),
                                  do_sample=False, pad_token_id=tok.pad_token_id)
            cont = gen[0, cut:].tolist()
            run = longest_match_run(cont, gold)
            matched = np.mean([a == b for a, b in zip(cont, gold)])
            frac_matched.append(matched)
            long_runs.append(run)
            if best is None or run > best[0]:
                best = (run, tok.decode(ids[:cut]), tok.decode(gold[:run]))
        return frac_matched, long_runs, best

    can_fm, can_run, can_best = run_set(CANONICAL)
    nov_fm, nov_run, _ = run_set(NOVEL)
    print(f"\n[{name}]  ({model.num_parameters() / 1e6:.0f}M params)", flush=True)
    print(f"  canonical: mean verbatim {np.mean(can_fm):.2f}  "
          f"mean exact-run {np.mean(can_run):.1f} tokens  "
          f"regurgitated(>=20 tok): {np.mean([r >= 20 for r in can_run]):.2f}",
          flush=True)
    print(f"  novel    : mean verbatim {np.mean(nov_fm):.2f}  "
          f"mean exact-run {np.mean(nov_run):.1f} tokens", flush=True)
    if can_best:
        print(f"  longest verbatim run: {can_best[0]} tokens", flush=True)
        print(f"    prompt: ...{can_best[1][-60:]!r}", flush=True)
        print(f"    model continued (verbatim): {can_best[2]!r}", flush=True)

    return {
        "canonical_verbatim": float(np.mean(can_fm)),
        "canonical_run": float(np.mean(can_run)),
        "canonical_regurg": float(np.mean([r >= 20 for r in can_run])),
        "novel_verbatim": float(np.mean(nov_fm)),
        "novel_run": float(np.mean(nov_run)),
    }, (name, can_run, nov_run)


def short_label(text, k=5):
    words = text.replace("\n", " ").split()
    lab = " ".join(words[:k])
    return (lab[:34] + "…") if len(lab) > 35 else lab + "…"


@torch.no_grad()
def per_text_breakdown(name, prefix_frac=0.5):
    """Per-text longest verbatim run for gpt2-large: which famous openings does
    the model finish word for word? Contrasted with the novel-text mean."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(12)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()

    def run_one(text):
        ids = tok(text, return_tensors="pt").input_ids[0]
        cut = max(4, int(prefix_frac * len(ids)))
        gold = ids[cut:].tolist()
        if len(gold) < 4:
            return 0
        gen = model.generate(ids[:cut].unsqueeze(0), max_new_tokens=len(gold),
                             do_sample=False, pad_token_id=tok.pad_token_id)
        return longest_match_run(gen[0, cut:].tolist(), gold)

    rows = [(short_label(t), run_one(t)) for t in CANONICAL]
    novel_mean = float(np.mean([run_one(t) for t in NOVEL]))
    rows.sort(key=lambda r: r[1])
    for lab, run in rows:
        print(f"  {run:3d} tok  {lab}", flush=True)
    print(f"  novel mean: {novel_mean:.1f} tok", flush=True)
    return rows, novel_mean


def run():
    results, dists = [], []
    for name in MODELS:
        r, dist = probe_model(name)
        r["model"] = name
        results.append(r)
        dists.append(dist)

    print("\nper-text breakdown (gpt2-large):", flush=True)
    rows, novel_mean = per_text_breakdown("gpt2-large")

    with open(OUT / "memorize.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "canonical_verbatim", "canonical_run",
                    "canonical_regurg", "novel_verbatim", "novel_run"])
        for r in results:
            w.writerow([r["model"], f"{r['canonical_verbatim']:.4f}",
                        f"{r['canonical_run']:.2f}", f"{r['canonical_regurg']:.4f}",
                        f"{r['novel_verbatim']:.4f}", f"{r['novel_run']:.2f}"])
    with open(OUT / "per_text.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "run"])
        for lab, run in rows:
            w.writerow([lab, run])
        w.writerow(["NOVEL (mean)", f"{novel_mean:.2f}"])
    plot()


def plot():
    rows = list(csv.DictReader(open(OUT / "memorize.csv")))
    models = [r["model"] for r in rows]

    fig, ax = ps.new_axes(width=7.0, height=4.4)
    x = np.arange(len(models))
    w = 0.36
    can = [float(r["canonical_run"]) for r in rows]
    nov = [float(r["novel_run"]) for r in rows]
    ax.bar(x - w / 2, can, w, color=ps.SERIES[2],
           label="canonical (likely in training data)")
    ax.bar(x + w / 2, nov, w, color=ps.SERIES[0], label="novel (never seen)")
    for xi, v in zip(x - w / 2, can):
        ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color=ps.INK)
    for xi, v in zip(x + w / 2, nov):
        ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color=ps.INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({'124M' if m == 'gpt2' else '774M'})"
                        for m in models])
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ps.finish(fig, ax, "Verbatim regurgitation grows with scale, and only on seen text",
              "", "mean longest exact-match run (tokens)", OUT / "memorize.png")

    pt = list(csv.DictReader(open(OUT / "per_text.csv")))
    labels = [r["label"] for r in pt]
    runs = [float(r["run"]) for r in pt]
    colors = [ps.SERIES[0] if lab.startswith("NOVEL") else ps.SERIES[2]
              for lab in labels]
    fig, ax = ps.new_axes(width=7.6, height=6.2)
    ax.barh(range(len(labels)), runs, color=colors, height=0.72)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i, v in enumerate(runs):
        ax.text(v + 0.4, i, f"{v:.0f}", va="center", fontsize=8, color=ps.INK)
    ax.set_xlim(0, max(runs) * 1.15 + 1)
    ps.finish(fig, ax,
              "Which openings gpt2-large finishes verbatim (given the first half)",
              "longest exact-match run (tokens)", "", OUT / "per_text.png")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    else:
        run()
