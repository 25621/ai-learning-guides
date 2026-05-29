# CoT vs. Direct on GSM8K

---

> Ask the model to show its work, and it gets the answer right far more often.

---

## Key Insight

[Chain-of-thought (CoT)](/shared/glossary/#cot) prompting asks the model to write out its reasoning step by step before giving a final answer, instead of replying right away. This project measures the accuracy gain from that single change on [GSM8K](/shared/glossary/#gsm8k) grade-school math problems.

## Why This Matters

The intermediate reasoning tokens give the model more room to compute and a place to lay out partial results. The same model — with no retraining at all — solves much harder problems just by being told to think out loud first.
