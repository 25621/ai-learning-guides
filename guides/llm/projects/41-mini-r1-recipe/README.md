# Mini R1 Recipe

---

> Reward only correct answers, and watch the model teach itself to reason.

---

## Key Insight

This project reproduces a small version of the DeepSeek-R1 recipe: lightly fine-tune ([SFT](/shared/glossary/#sft)) a [base model](/shared/glossary/#base-model) on a few reasoning traces, then run [GRPO](/shared/glossary/#grpo) with a simple "is the answer correct?" [verifier](/shared/glossary/#verifier) — a form of [RLVR](/shared/glossary/#rlvr) — on math problems.

## Why This Matters

With nothing but a correctness signal, the model spontaneously grows long [chain-of-thought](/shared/glossary/#cot) habits like backtracking and self-checking. This emergence is the core discovery behind modern reasoning models.
