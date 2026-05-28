# Agent on a [Sandbox](/shared/glossary/#sandbox) Task

---

> An agent is just a model in a loop — and loops are where things go wrong.

---

## Key Insight

An [agent](/shared/glossary/#agent) is an LLM placed in a loop: it plans, picks a tool, acts, observes the result, and repeats until the task is done. This project builds a tiny [ReAct](/shared/glossary/#react)-style agent on a deterministic task and runs it across many [seeds](/shared/glossary/#seed) to measure how reliably it actually succeeds.

## Why This Matters

A single correct answer is easy; staying on track across many steps is hard, because a small per-step error rate compounds into frequent whole-task failure — which is why agents must be judged over many runs, not on one lucky success.
