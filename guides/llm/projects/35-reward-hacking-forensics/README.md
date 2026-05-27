# Reward-Hacking Forensics

---

> When the score goes up but the answers get worse, find out which part broke.

---

## Key Insight

This project deliberately trains a [reward-hacked](/shared/glossary/#reward-hacking) model, then traces the failure back to its real source — the [reward model](/shared/glossary/#reward-model), the [KL](/shared/glossary/#kl-divergence) penalty (β), or the [rollout](/shared/glossary/#rollout) distribution. [Forensics](/shared/glossary/#forensics) here means working backward from the broken behavior to the cause instead of guessing.

## Why This Matters

Reward hacking is the most common way [RLHF](/shared/glossary/#rlhf) goes wrong, and the symptom rarely points straight at the cause. Learning to diagnose it systematically is what separates a frustrating week from a quick fix.
