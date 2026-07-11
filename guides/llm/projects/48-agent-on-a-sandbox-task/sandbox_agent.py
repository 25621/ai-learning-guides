"""Project 48 — Agent on a sandbox task: a tiny ReAct loop on a spreadsheet.

The environment is a 12-cell spreadsheet (a1..d3, random 2-digit values every
seed). The agent has three tools plus a stop action:

    G:a1;      read a cell          -> R:57;
    C:57+82;   calculator           -> R:139;
    S:c3=139;  write a cell         -> R:ok;
    A:done;    declare finished

Tasks come in four sizes, from "copy one cell" (3 agent actions) to "sum four
cells into a target" (9 actions). Success is judged on the *environment*: the
target cell must hold the right number after the episode.

The point of the project is the reliability curve: if the agent's chance of
producing the right next action is p, a task needing h actions succeeds
around p^h — per-step polish decays exponentially with horizon, and sampling
temperature turns the decay knob.
"""

import csv
import os
import random
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "47-tool-using-chatbot"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))

import plot_style as ps  # noqa: E402
import tool_lib as tl  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

BLOCK = 160
CELLS = [c + r for c in "abcd" for r in "123"]
TASKS = {"cp": 3, "sum2": 5, "sum3": 7, "sum4": 9}   # -> agent actions
STEPS = 1600
N_EVAL = 75
TEMPS = [0.0, 1.3]
_SET_RE = re.compile(r"([a-d][1-3])=(\d{1,3})")


def make_task(rng, kind):
    grid = {c: rng.randint(10, 99) for c in CELLS}
    n_src = 1 if kind == "cp" else int(kind[-1])
    cells = rng.sample(CELLS, n_src + 1)
    src, tgt = cells[:-1], cells[-1]
    if kind == "cp":
        q = f"Q:cp({src[0]},{tgt})?"
        expected = grid[src[0]]
    else:
        q = f"Q:{tgt}={'+'.join(src)}?"
        expected = sum(grid[c] for c in src)
    return {"kind": kind, "q": q, "grid": grid, "src": src, "tgt": tgt,
            "expected": expected}


def expert_pieces(t):
    """Gold ReAct trace: read sources, fold with the calculator, write."""
    grid, src, tgt = t["grid"], t["src"], t["tgt"]
    pieces = [(t["q"], 0)]
    vals = [grid[c] for c in src]
    for c, v in zip(src, vals):
        pieces += [(f"G:{c};", 1), (f"R:{v};", 0)]
    acc = vals[0]
    for v in vals[1:]:
        pieces += [(f"C:{acc}+{v};", 1), (f"R:{acc + v};", 0)]
        acc += v
    pieces += [(f"S:{tgt}={acc};", 1), ("R:ok;", 0), ("A:done;", 1)]
    return pieces


def sheet_traces(bs, rng):
    out = []
    for _ in range(bs):
        t = make_task(rng, rng.choice(list(TASKS)))
        out.append(tl.pieces_to_trace(expert_pieces(t)))
    return out


def sheet_step(state, seg):
    grid = state["grid"]
    if seg.startswith("A:"):
        state["answered"] = True
        return None, True
    if seg.startswith("G:") and seg[2:] in grid:
        return f"R:{grid[seg[2:]]};", False
    if seg.startswith("C:"):
        m = tl._CALC_RE.fullmatch(seg[2:])
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            return f"R:{a + b if op == '+' else a * b};", False
    if seg.startswith("S:"):
        m = _SET_RE.fullmatch(seg[2:])
        if m:
            grid[m.group(1)] = int(m.group(2))
            return "R:ok;", False
    state["malformed"] += 1
    return "R:?;", False


def evaluate(model, kind, n, temperature, seed=1234):
    rng = random.Random(seed)
    tasks = [make_task(rng, kind) for _ in range(n)]
    states = [{"grid": dict(t["grid"]), "answered": False, "malformed": 0}
              for t in tasks]
    tl.run_episodes(model, [t["q"] for t in tasks], sheet_step, states,
                    block=BLOCK, max_model_tokens=90,
                    temperature=temperature)
    ok = [st["answered"] and st["grid"][t["tgt"]] == t["expected"]
          for t, st in zip(tasks, states)]
    return float(np.mean(ok))


@torch.no_grad()
def per_action_reliability(model, temperature, n=150, seed=99):
    """Teacher-forced p: probability the model reproduces each expert action.

    One batched forward over expert traces; a segment's success probability
    is the product of its tokens' model probabilities (argmax -> 0/1 at
    temperature 0). Returns the mean over all expert actions."""
    rng = random.Random(seed)
    probs = []
    traces = []
    for _ in range(n):
        t = make_task(rng, rng.choice(list(TASKS)))
        traces.append(tl.pieces_to_trace(expert_pieces(t)))
    for lo in range(0, n, 64):
        batch = traces[lo:lo + 64]
        T = max(len(ids) for ids, _ in batch)
        x = torch.zeros(len(batch), T, dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            x[i, :len(ids)] = torch.tensor(ids)
        logits, _ = model(x[:, :-1])
        for i, (ids, mask) in enumerate(batch):
            if temperature > 0:
                lp = F.log_softmax(logits[i] / temperature, dim=-1)
            else:
                lp = (logits[i].argmax(-1).unsqueeze(-1) ==
                      torch.arange(tl.VOCAB)).float().clamp(min=1e-9).log()
            seg_lp, in_seg = 0.0, False
            for pos in range(1, len(ids)):
                if mask[pos]:
                    seg_lp += float(lp[pos - 1, ids[pos]])
                    in_seg = True
                    if ids[pos] == tl.SEG_END:
                        probs.append(np.exp(seg_lp))
                        seg_lp, in_seg = 0.0, False
            assert not in_seg
    return float(np.mean(probs))


def main():
    model = tl.new_model(block=BLOCK, seed=0)
    ckpt = os.path.join(HERE, "checkpoints", "agent_sft.pt")
    if os.path.exists(ckpt):
        print("== loading cached SFT checkpoint ==", flush=True)
        model.load_state_dict(torch.load(ckpt))
    else:
        print("== SFT on expert ReAct traces ==", flush=True)
        tl.train_sft(model, sheet_traces, STEPS, block=BLOCK, tag="agent ")
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save(model.state_dict(), ckpt)
    model.eval()

    rows, curves = [], {}
    for temp in TEMPS:
        p = per_action_reliability(model, temp)
        succ = {}
        for kind, h in TASKS.items():
            succ[h] = evaluate(model, kind, N_EVAL, temp)
            print(f"temp {temp}: {kind} (h={h}) success {succ[h]:.2f} "
                  f"(p^h predicts {p ** h:.2f})", flush=True)
            rows.append([temp, kind, h, f"{succ[h]:.3f}", f"{p:.4f}",
                         f"{p ** h:.3f}"])
        curves[temp] = (p, succ)
        print(f"temp {temp}: per-action reliability p = {p:.4f}")

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temperature", "task", "horizon_actions", "success",
                    "per_action_p", "p^h"])
        w.writerows(rows)

    hs = sorted(TASKS.values())
    fig, ax = ps.new_axes(7.6, 4.4)
    hh = np.linspace(min(hs), max(hs), 100)
    for i, temp in enumerate(TEMPS):
        p, succ = curves[temp]
        ax.plot(hh, p ** hh, "--", color=ps.SERIES[i], linewidth=1.4,
                alpha=0.7)
        ax.plot(hs, [succ[h] for h in hs], "o-", color=ps.SERIES[i],
                linewidth=2, label=f"temp {temp} (p={p:.3f}); dashed: p^h")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(hs)
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Per-step reliability compounds over the horizon",
              "actions required (horizon h)", "task success rate",
              os.path.join(OUT, "sandbox_agent.png"))


if __name__ == "__main__":
    main()
