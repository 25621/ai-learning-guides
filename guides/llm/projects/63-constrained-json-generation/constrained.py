"""Constrained JSON generation: measure what the schema guarantees, and what it costs.

Same model, same prompts, two decoders:

  free         ordinary greedy decoding. Whatever the model says is what you get.
  constrained  the FSM from fsm.py masks every token that would break the schema.

We report the three numbers that decide whether you turn this on in production:
the fraction of outputs a JSON parser accepts, whether the *content* got worse,
and the latency the mask adds.

Run:  python3 constrained.py           (~7 min)
      python3 constrained.py --plot    (redraw from outputs/results.json)
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import fsm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-train-a-bpe-from-scratch"))
import plot_style as ps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoints")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_ITEMS = 30
MAX_NEW = 48
torch.set_num_threads(12)

NAMES = ["Ada Lovelace", "Grace Hopper", "Alan Turing", "Katherine Johnson",
         "Linus Torvalds", "Barbara Liskov", "Ken Thompson", "Radia Perlman",
         "Donald Knuth", "Margaret Hamilton"]
CITIES = ["Lisbon", "Osaka", "Nairobi", "Toronto", "Helsinki", "Bogota",
          "Jakarta", "Dublin", "Seoul", "Lima"]
FORMS = [
    "{name} is {age} years old and lives in {city}.",
    "Our guest {name}, aged {age}, flew in from {city} this morning.",
    "{city} resident {name} celebrated turning {age} last week.",
    "At {age}, {name} relocated to {city} for a new job.",
]


def dataset(n=N_ITEMS, seed=0):
    rng = random.Random(seed)
    items = []
    for i in range(n):
        name, city = rng.choice(NAMES), rng.choice(CITIES)
        age = rng.randint(19, 74)
        text = rng.choice(FORMS).format(name=name, age=age, city=city)
        items.append(dict(text=text, name=name, age=age, city=city))
    return items


def prompt_for(tok, item):
    body = ('Extract the person\'s name, age and city from the sentence and reply '
            'with JSON using the keys "name", "age", "city". Reply with JSON only.\n\n'
            f'Sentence: {item["text"]}')
    return tok.apply_chat_template([{"role": "user", "content": body}],
                                   tokenize=False, add_generation_prompt=True)


# ------------------------------------------------------------------- decoding
@torch.no_grad()
def generate_free(model, tok, prompt, max_new=MAX_NEW):
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model(ids, use_cache=True)
    past = out.past_key_values
    toks, t_mask = [], 0.0
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(max_new):
        tid = int(nxt)
        if tid == tok.eos_token_id:
            break
        toks.append(tid)
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    return tok.decode(toks), len(toks), t_mask


@torch.no_grad()
def generate_constrained(model, tok, prompt, masker, max_new=MAX_NEW):
    """Identical loop, plus one masked_fill before the argmax."""
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model(ids, use_cache=True)
    past = out.past_key_values
    state = (0, 0)
    toks, t_mask = [], 0.0

    logits = out.logits[:, -1]
    for _ in range(max_new):
        t0 = time.perf_counter()
        logits = masker.apply(logits, state)          # <- the entire intervention
        t_mask += time.perf_counter() - t0
        tid = int(logits.argmax(-1))
        if tid == tok.eos_token_id:
            break
        toks.append(tid)
        state = masker.advance(state, tid)
        out = model(torch.tensor([[tid]]), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1]
    return tok.decode(toks), len(toks), t_mask


# ------------------------------------------------------------------- scoring
def parse(text):
    """Strict: does a JSON parser accept exactly what the model emitted?"""
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_lenient(text):
    """Be generous: strip markdown fences and trailing prose, then retry."""
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return None
    return None


def valid_schema(obj):
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"name", "age", "city"}:
        return False
    return isinstance(obj["name"], str) and isinstance(obj["age"], int) \
        and isinstance(obj["city"], str)


def field_scores(obj, item):
    """Per-field correctness — which field the decoder actually got wrong."""
    if not obj:
        return dict(name=0.0, age=0.0, city=0.0)
    name = float(str(obj.get("name", "")).strip().lower() == item["name"].lower())
    city = float(str(obj.get("city", "")).strip().lower() == item["city"].lower())
    try:
        age = float(int(obj.get("age")) == item["age"])
    except Exception:
        age = 0.0
    return dict(name=name, age=age, city=city)


def content_score(obj, item):
    """Did the extraction get the facts right? (only defined for parseable output)"""
    f = field_scores(obj, item)
    return (f["name"] + f["age"] + f["city"]) / 3


def run(mode, model, tok, items, masker=None):
    recs, t_total, t_mask_total, ntok = [], 0.0, 0.0, 0
    for it in items:
        p = prompt_for(tok, it)
        t0 = time.perf_counter()
        if mode == "free":
            text, n, tm = generate_free(model, tok, p)
        else:
            text, n, tm = generate_constrained(model, tok, p, masker)
        t_total += time.perf_counter() - t0
        t_mask_total += tm
        ntok += n

        strict, lenient = parse(text), parse_lenient(text)
        obj = strict if strict is not None else lenient
        recs.append(dict(text=text, strict=strict is not None and valid_schema(strict),
                         lenient=lenient is not None and valid_schema(lenient),
                         content=content_score(obj, it), **field_scores(obj, it)))

    n = len(items)
    row = dict(mode=mode,
               strict=sum(r["strict"] for r in recs) / n,
               lenient=sum(r["lenient"] for r in recs) / n,
               content=sum(r["content"] for r in recs) / n,
               name=sum(r["name"] for r in recs) / n,
               age=sum(r["age"] for r in recs) / n,
               city=sum(r["city"] for r in recs) / n,
               ms_per_token=t_total / ntok * 1000,
               mask_ms_per_token=t_mask_total / ntok * 1000,
               tokens=ntok / n, wall=t_total)
    print(f"  {mode:11s} | schema-valid (strict) {row['strict']:5.0%} | "
          f"(after cleanup) {row['lenient']:5.0%} | fields correct {row['content']:5.1%} "
          f"(name {row['name']:.0%} age {row['age']:.0%} city {row['city']:.0%}) | "
          f"{row['ms_per_token']:5.1f} ms/tok (mask {row['mask_ms_per_token']:.2f}) | "
          f"{row['tokens']:.0f} tokens", flush=True)
    return row, recs


# -------------------------------------------------------------------- report
def plot(rows, compile_s, examples):
    free = [r for r in rows if r["mode"] == "free"][0]
    con = [r for r in rows if r["mode"] == "constrained"][0]

    # validity + content
    fig, ax = ps.new_axes(7.4, 4.2)
    groups = ["parses as JSON\n(strict)", "parses after\ncleanup", "fields correct"]
    fv = [free["strict"], free["lenient"], free["content"]]
    cv = [con["strict"], con["lenient"], con["content"]]
    x = range(len(groups))
    ax.bar([i - 0.19 for i in x], fv, 0.36, color=ps.SERIES[2], label="free decoding",
           zorder=3)
    ax.bar([i + 0.19 for i in x], cv, 0.36, color=ps.SERIES[0], label="constrained",
           zorder=3)
    for i, (a, b) in enumerate(zip(fv, cv)):
        ax.annotate(f"{a:.0%}", xy=(i - 0.19, a), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=ps.INK_SECONDARY, fontsize=9)
        ax.annotate(f"{b:.0%}", xy=(i + 0.19, b), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=9)
    ps.finish(fig, ax, "The schema is guaranteed; the content is not",
              "", "fraction of 30 extractions", os.path.join(OUT, "validity.png"))

    # cost
    fig, ax = ps.new_axes(7.2, 4.0)
    labels = ["free decoding", "constrained\n(model forward)", "constrained\n(logit mask)"]
    vals = [free["ms_per_token"], con["ms_per_token"] - con["mask_ms_per_token"],
            con["mask_ms_per_token"]]
    ax.bar(labels, vals, color=[ps.SERIES[2], ps.SERIES[0], ps.SERIES[1]],
           width=0.55, zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f} ms", xy=(i, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=ps.INK_SECONDARY, fontsize=9)
    ax.annotate(f"the mask is {con['mask_ms_per_token']/free['ms_per_token']:.1%} of a "
                f"decode step\n(plus a one-off {compile_s:.0f}s index compile)",
                xy=(2, vals[2]), xytext=(-10, 40), textcoords="offset points",
                ha="center", color=ps.INK_SECONDARY, fontsize=9)
    ps.finish(fig, ax, "What constraining costs per token",
              "", "milliseconds per generated token", os.path.join(OUT, "overhead.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
    cache = os.path.join(CKPT, "results.json")

    if args.plot:
        d = json.load(open(cache))
        plot(d["rows"], d["compile_s"], d["examples"])
        return

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    items = dataset()

    # Compile the index once, up front — the way a real constrained-decoding
    # library does — so the per-token numbers below measure decoding, not startup.
    t0 = time.perf_counter()
    masker = fsm.TokenMasker(tok, tok.eos_token_id,
                             logit_width=model.config.vocab_size)
    n_states = masker.precompile()
    compile_s = time.perf_counter() - t0
    print(f"vocabulary {masker.n_tokens:,} tokens ({masker.vocab_size:,} logit rows) | "
          f"index compiled: {n_states} states in {compile_s:.1f}s (one-off, at startup)\n")

    rows, all_recs = [], {}
    for mode in ("free", "constrained"):
        row, recs = run(mode, model, tok, items, masker)
        rows.append(row); all_recs[mode] = recs

    total_compile = compile_s
    print(f"\n  FSM states compiled: {masker.compiled_states} "
          f"(cached and reused across all {len(items)} requests)")

    examples = [dict(sentence=items[i]["text"], free=all_recs["free"][i]["text"],
                     constrained=all_recs["constrained"][i]["text"]) for i in range(3)]
    json.dump(dict(rows=rows, compile_s=total_compile, examples=examples,
                   states=masker.compiled_states), open(cache, "w"), indent=1)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "examples.txt"), "w") as f:
        for e in examples:
            f.write(f"sentence     : {e['sentence']}\n")
            f.write(f"free         : {e['free']!r}\n")
            f.write(f"constrained  : {e['constrained']!r}\n\n")
    plot(rows, total_compile, examples)


if __name__ == "__main__":
    main()
