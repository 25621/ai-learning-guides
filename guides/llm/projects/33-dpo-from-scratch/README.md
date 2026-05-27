# DPO from Scratch

---

> Get the result of RLHF without running any of the reinforcement learning.

---

## Key Insight

This project implements [DPO](/shared/glossary/#dpo) by hand and checks it against the reference loss in a library like TRL. DPO skips the [reward model](/shared/glossary/#reward-model) and the [PPO](/shared/glossary/#ppo) loop entirely, collapsing preference learning into a single supervised [loss](/shared/glossary/#loss-function) on (chosen, rejected) answer pairs.

## Why This Matters

DPO made alignment dramatically simpler — no reward model, no [rollouts](/shared/glossary/#rollout), just two models and a loss. It became a default open-source recipe because it captures much of [RLHF](/shared/glossary/#rlhf)'s benefit with a fraction of the moving parts.
