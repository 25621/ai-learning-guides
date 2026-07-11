"""Project 49 — SWE-style coding agent: file editor + test runner in a loop.

A synthetic bug benchmark stands in for SWE-bench: each "issue" is a repo
with one file

    def f(x):
        return x+6        # <- planted one-line bug (wrong op or constant)

and a test suite of two asserts (`f(3)=12`, `f(5)=20`). The agent sees the
failing tests plus the current code and acts through two real tools:

    E:return x*4;   the *editor* — rewrites the return line of calc.py on disk
    (after every edit the harness — the CI — reruns the tests on the file and
     reports `R:pass;` or `R:fail got 9,15;`)
    A:done;         stop

Fixing the bug requires actual inference from the tests: which (op, constant)
reproduces both expected outputs? Tasks are generated to be unambiguous.

Two agents, same budget, differ only in training traces:
  * plain     — perfect traces only (edit -> pass -> done)
  * recovery  — 40% of traces open with a *wrong* first edit whose tokens are
                LOSS-MASKED (an environment-forced error, so the model never
                learns to imitate it), followed by the failing test report and
                the correct second edit. The model learns what to do *after*
                a failure without learning to fail.
"""

import csv
import os
import random
import re
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "47-tool-using-chatbot"))
sys.path.insert(0, os.path.join(HERE, "..", "01-train-a-bpe-from-scratch"))

import plot_style as ps  # noqa: E402
import tool_lib as tl  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

BLOCK = 160
STEPS = 1200
N_EVAL = 200
MAX_EDITS = 3
OPS = "+-*"
_EDIT_RE = re.compile(r"return x([+*-])(\d{1,2})")


def apply(op, c, x):
    return x + c if op == "+" else x - c if op == "-" else x * c


def _rand_fix(rng):
    op = rng.choice(OPS)
    return op, rng.randint(2, 9 if op == "*" else 19)


def make_issue(rng):
    """One synthetic bug: true (op,c), two unambiguous tests, a buggy line."""
    while True:
        op, c = _rand_fix(rng)
        xs = rng.sample(range(20, 40) if op == "-" else range(2, 10), 2)
        tests = [(x, apply(op, c, x)) for x in xs]
        if any(any(all(apply(o2, c2, x) == e for x, e in tests)
                   for c2 in range(0, 20))
               for o2 in OPS.replace(op, "")):
            continue  # ambiguous — another (op, c) explains the tests too
        while True:
            bop, bc = _rand_fix(rng)
            if (bop, bc) != (op, c) and any(
                    apply(bop, bc, x) != e for x, e in tests):
                break
        return {"op": op, "c": c, "tests": tests, "bop": bop, "bc": bc}


def prompt_str(iss):
    (x1, e1), (x2, e2) = iss["tests"]
    return (f"Q:f({x1})={e1},f({x2})={e2};"
            f"code return x{iss['bop']}{iss['bc']};")


def fail_reply(iss, op, c):
    got = ",".join(str(apply(op, c, x)) for x, _ in iss["tests"])
    return f"R:fail got {got};"


def issue_traces(recovery_frac):
    def sample(bs, rng):
        out = []
        for _ in range(bs):
            iss = make_issue(rng)
            fix = f"E:return x{iss['op']}{iss['c']};"
            pieces = [(prompt_str(iss), 0)]
            if rng.random() < recovery_frac:
                # environment-forced wrong first edit, loss-masked (0)
                while True:
                    wop, wc = _rand_fix(rng)
                    if (wop, wc) != (iss["op"], iss["c"]):
                        break
                pieces += [(f"E:return x{wop}{wc};", 0),
                           (fail_reply(iss, wop, wc), 0)]
            pieces += [(fix, 1), ("R:pass;", 0), ("A:done;", 1)]
            out.append(tl.pieces_to_trace(pieces))
        return out
    return sample


# ------------------------------------------------------- the real harness

def write_repo(ws, iss):
    with open(os.path.join(ws, "calc.py"), "w") as f:
        f.write(f"def f(x):\n    return x{iss['bop']}{iss['bc']}\n")


def run_tests(ws, iss):
    """CI step: exec the file as it exists on disk, run the asserts."""
    ns = {}
    with open(os.path.join(ws, "calc.py")) as f:
        try:
            exec(f.read(), ns)
            got = [ns["f"](x) for x, _ in iss["tests"]]
        except Exception:
            return False, "R:fail got error;"
    if all(g == e for g, (_, e) in zip(got, iss["tests"])):
        return True, "R:pass;"
    return False, f"R:fail got {','.join(str(g) for g in got)};"


def issue_step(state, seg):
    iss, ws = state["iss"], state["ws"]
    if seg.startswith("A:"):
        state["answered"] = True
        return None, True
    if seg.startswith("E:"):
        state["edits"] += 1
        state["edit_log"].append(seg[2:])
        m = _EDIT_RE.fullmatch(seg[2:])
        if m:
            with open(os.path.join(ws, "calc.py"), "w") as f:
                f.write(f"def f(x):\n    return x{m.group(1)}{m.group(2)}\n")
        passed, reply = run_tests(ws, iss)
        if not state["solved_at"] and passed:
            state["solved_at"] = state["edits"]
        if state["edits"] >= MAX_EDITS and not passed:
            return None, True  # budget exhausted
        return reply, False
    state["malformed"] += 1
    return "R:?;", False


def evaluate(model, n, seed=1234):
    rng = random.Random(seed)
    issues = [make_issue(rng) for _ in range(n)]
    ws = tempfile.mkdtemp(prefix="swe_")
    states = []
    for iss in issues:
        d = tempfile.mkdtemp(dir=ws)
        write_repo(d, iss)
        states.append({"iss": iss, "ws": d, "edits": 0, "solved_at": 0,
                       "answered": False, "malformed": 0, "edit_log": []})
    seqs, _ = tl.run_episodes(model, [prompt_str(i) for i in issues],
                              issue_step, states, block=BLOCK,
                              max_model_tokens=60)
    solved_at = np.array([st["solved_at"] for st in states])
    first_failed = solved_at != 1
    rec = ((solved_at > 1).sum() / max(first_failed.sum(), 1))
    # after a failed first edit, does the agent just repeat itself?
    repeats, retried = 0, 0
    for st in states:
        if st["solved_at"] != 1 and len(st["edit_log"]) >= 2:
            retried += 1
            repeats += st["edit_log"][1] == st["edit_log"][0]
    return {
        "repeat_rate": repeats / max(retried, 1),
        "solved@k": {k: float((np.logical_and(solved_at > 0,
                                              solved_at <= k)).mean())
                     for k in range(1, MAX_EDITS + 1)},
        "recovery_rate": float(rec),
        "n_first_failed": int(first_failed.sum()),
        "states": states, "seqs": seqs,
    }


def main():
    models = {}
    for name, frac in [("plain", 0.0), ("recovery-trained", 0.4)]:
        print(f"== SFT: {name} agent ==", flush=True)
        m = tl.new_model(block=BLOCK, n_embd=96, seed=0)
        tl.train_sft(m, issue_traces(frac), STEPS, block=BLOCK,
                     tag=f"{name} ")
        m.eval()
        models[name] = m

    rows, results = [], {}
    for name, m in models.items():
        r = evaluate(m, N_EVAL)
        results[name] = r
        sk = r["solved@k"]
        print(f"{name:17s} solved@1 {sk[1]:.3f}  solved@2 {sk[2]:.3f}  "
              f"solved@3 {sk[3]:.3f}  recovery {r['recovery_rate']:.2f} "
              f"(of {r['n_first_failed']} first-edit misses)  "
              f"repeat-after-fail {r['repeat_rate']:.2f}")
        rows.append([name] + [f"{sk[k]:.3f}" for k in sk] +
                    [f"{r['recovery_rate']:.3f}", r["n_first_failed"],
                     f"{r['repeat_rate']:.3f}"])

    print("== sample episodes (recovery-trained) ==")
    r = results["recovery-trained"]
    shown = 0
    for st, sq in zip(r["states"], r["seqs"]):
        if st["solved_at"] > 1 and shown < 3:
            print("  " + tl.decode(sq).replace("\n", " "))
            shown += 1

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "solved@1", "solved@2", "solved@3",
                    "recovery_rate", "n_first_failed", "repeat_after_fail"])
        w.writerows(rows)

    fig, ax = ps.new_axes(7.2, 4.2)
    ks = list(range(1, MAX_EDITS + 1))
    for i, name in enumerate(models):
        vals = [results[name]["solved@k"][k] for k in ks]
        ax.plot(ks, vals, "o-", color=ps.SERIES[i], linewidth=2, label=name)
        for k, v in zip(ks, vals):
            ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8,
                        color=ps.INK_SECONDARY)
    ax.set_xticks(ks)
    ax.set_ylim(0.3, 0.65)
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY,
              loc="lower right")
    ps.finish(fig, ax, "Issues fixed vs edit budget (200 synthetic bugs)",
              "edit budget", "fraction of issues solved",
              os.path.join(OUT, "swe_agent.png"))


if __name__ == "__main__":
    main()
