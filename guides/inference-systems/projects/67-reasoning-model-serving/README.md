# Reasoning-Model Serving

---

> A [reasoning model](/shared/glossary/#reasoning-model) can spend 5 tokens or 5,000 on the same prompt — serving it well is mostly about taming that uncertainty.

---

## Key Insight

This project serves a long-[chain-of-thought](/shared/glossary/#cot) [reasoning model](/shared/glossary/#reasoning-model), measures how wildly its output length swings from one request to the next, and then adds a [thinking-budget](/shared/glossary/#thinking-budget) knob that caps how long the model is allowed to think before it must answer. Watching the output-length distribution makes the core problem visible: unlike a chat model, you cannot guess how much work a single request will be.

## Why This Matters

Output-length variance is what breaks naive capacity planning for reasoning models: a handful of hard prompts can each generate 10× the tokens of a normal reply, blowing up [latency](/shared/glossary/#latency) and [cost](/shared/glossary/#cost-per-million-tokens) for everyone sharing the GPU. A thinking budget gives you a direct dial to trade accuracy for predictable cost and tail latency — the single most useful control when serving this class of model.
