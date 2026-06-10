# GRPO from Scratch

## Key Insight

[Group Relative Policy Optimization (GRPO)](/shared/glossary/#grpo) is DeepSeek's slimmed-down [PPO](/shared/glossary/#ppo): it deletes the separate critic (the [value function](/shared/glossary/#value-function) network PPO uses as a baseline) and replaces it with a much cheaper trick — for each prompt, sample a *group* of completions, score them all, and use each completion's [advantage](/shared/glossary/#advantage) *relative to the group's average* as the learning signal. This project implements GRPO from scratch on a small math task (GSM8K-style), where the score can come straight from a [verifier](/shared/glossary/#verifier) checking the final answer instead of a learned [reward model](/shared/glossary/#reward-model). Why it matters: dropping the critic roughly halves the memory and compute of RL fine-tuning, and this simplicity is why GRPO became the backbone of recent [reasoning-model](/shared/glossary/#reasoning-model) training.
