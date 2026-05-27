# GRPO on a Math Task

---

> Try a problem eight times, then learn from the tries that checked out.

---

## Key Insight

This project trains a model with [GRPO](/shared/glossary/#grpo) on [GSM8K](/shared/glossary/#gsm8k) grade-school math problems, using exact-answer matching as the [verifier](/shared/glossary/#verifier) — a form of [RLVR](/shared/glossary/#rlvr). For each problem the model samples a group of answers, and each answer is rewarded by how its score compares to the group's average.

## Why This Matters

GRPO drops both the [reward model](/shared/glossary/#reward-model) and the [value network](/shared/glossary/#value-network), making reasoning-focused RL cheap enough to run widely. It is the backbone of recent reasoning models like DeepSeek-R1.
