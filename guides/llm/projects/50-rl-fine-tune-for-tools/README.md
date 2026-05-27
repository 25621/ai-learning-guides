# RL Fine-Tune for Tools

---

> Reward the tool calls that work, and the model learns to use its tools well.

---

## Key Insight

Instead of only imitating recorded tool-call examples, this project trains the model with [GRPO](/shared/glossary/#grpo) using a [verifier](/shared/glossary/#verifier) that checks whether each tool call produced the expected output — a form of [RLVR](/shared/glossary/#rlvr) applied to tool use.

## Why This Matters

A verifiable success signal lets the model *practice* using its tools and learn from what actually works, rather than just copying traces — the frontier recipe for reliable tool-using [agents](/shared/glossary/#agent).
