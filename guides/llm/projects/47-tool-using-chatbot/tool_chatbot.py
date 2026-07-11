"""Project 47 — Tool-using chatbot: calculator + lookup via function calling.

Two models, identical architecture and training budget, on the same three
question kinds:

  tool model    — SFT on tool-call traces (`C:...;` / `L:...;`), the
                  orchestrator executes each call and appends `R:...;`
  direct model  — SFT on the same questions with the bare answer (`A:...;`)

The benchmark is rigged the way the real world is rigged: 'lookup' facts are
rolled fresh per episode (they exist only in the orchestrator's database, like
today's weather), and 'calc' needs exact 2-digit multiplication (which tiny
LMs are bad at). Tools turn both into copy jobs.
"""

import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))

import plot_style as ps  # noqa: E402
import tool_lib as tl  # noqa: E402

OUT = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoints")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

STEPS = 900
N_EVAL = 300


def main():
    print("== SFT: tool model (loss on C:/L:/A: segments only) ==",
          flush=True)
    tool_model = tl.new_model(seed=0)
    tl.train_sft(tool_model, tl.qa_traces(expert=True), STEPS, tag="tool ")
    torch.save(tool_model.state_dict(), os.path.join(CKPT, "tool_sft.pt"))

    print("== SFT: direct model (same budget, answers directly) ==",
          flush=True)
    direct_model = tl.new_model(seed=1)
    tl.train_sft(direct_model, tl.qa_traces(expert=False), STEPS,
                 tag="direct ")

    results = {}
    rows = []
    for name, model in [("direct", direct_model), ("tool", tool_model)]:
        for kind in tl.KINDS:
            r = tl.qa_eval(model, kind, N_EVAL)
            results[(name, kind)] = r
            print(f"{name:6s} {kind:8s} acc {r['acc']:.3f}  "
                  f"malformed {r['malformed']:.3f}  "
                  f"model-tokens/answer {r['model_tokens']:.1f}")
            rows.append([name, kind, f"{r['acc']:.3f}",
                         f"{r['malformed']:.3f}",
                         f"{r['model_tokens']:.1f}"])

    print("== sample transcripts (tool model) ==")
    for kind in tl.KINDS:
        r = results[("tool", kind)]
        print(f"  [{kind}] {tl.decode(r['seqs'][0])}")
    print("== sample hallucination (direct model, lookup) ==")
    r = results[("direct", "lookup")]
    s = r["states"][0]
    print(f"  {tl.decode(r['seqs'][0])}   (true value was {s['ans']})")

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "kind", "accuracy", "malformed_rate",
                    "model_tokens"])
        w.writerows(rows)

    fig, ax = ps.new_axes(7.6, 4.2)
    x = np.arange(len(tl.KINDS))
    for s, name in enumerate(["direct", "tool"]):
        vals = [results[(name, k)]["acc"] for k in tl.KINDS]
        off = (s - 0.5) * 0.38
        ax.bar(x + off, vals, width=0.38, color=ps.SERIES[s],
               label=f"{name} model")
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5,
                    color=ps.INK_SECONDARY)
    ax.set_xticks(x, [f"{k}\n(n={N_EVAL})" for k in tl.KINDS])
    ax.set_ylim(0, 1.06)
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Same model size, same training budget — only one "
              "can call tools", "question kind", "answer accuracy",
              os.path.join(OUT, "tool_chatbot.png"))


if __name__ == "__main__":
    main()
