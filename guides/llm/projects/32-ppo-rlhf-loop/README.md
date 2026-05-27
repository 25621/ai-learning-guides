# PPO RLHF Loop

---

> Chase the reward, but stay tied to the model you started from.

---

## Key Insight

This project wires together [SFT](/shared/glossary/#sft), a [reward model](/shared/glossary/#reward-model), and [PPO](/shared/glossary/#ppo) into a full [RLHF](/shared/glossary/#rlhf) loop, watching the reward climb and the [KL divergence](/shared/glossary/#kl-divergence) from the [reference model](/shared/glossary/#reference-model). Lowering the KL penalty (β) on purpose makes the [policy](/shared/glossary/#policy) start [reward hacking](/shared/glossary/#reward-hacking).

## Why This Matters

PPO-based RLHF is the classic recipe that first made chatbots both helpful and safe. Seeing the [KL](/shared/glossary/#kl-divergence) term hold the policy in check teaches the single most important knob in the entire [alignment stack](/shared/glossary/#alignment-stack).
