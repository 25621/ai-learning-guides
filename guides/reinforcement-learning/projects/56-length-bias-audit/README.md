# Length-Bias Audit

## Key Insight

[Length bias](/shared/glossary/#length-bias) is the well-documented tendency of [RLHF](/shared/glossary/#rlhf)-tuned models to grow more verbose over training — not because longer answers are genuinely better, but because [reward models](/shared/glossary/#reward-model) and human preference data quietly correlate length with quality, and the [policy](/shared/glossary/#policy) learns to exploit that correlation. This project plots the completion-length distributions of your [PPO](/shared/glossary/#ppo)- and [DPO](/shared/glossary/#dpo)-trained models before and after tuning to make the drift visible. Why it matters: length bias is a concrete, easy-to-measure instance of [reward hacking](/shared/glossary/#reward-hacking) — the model is maximizing a flawed proxy rather than true helpfulness — and spotting it is the first step toward the length-corrected losses (such as SimPO) that try to remove it.
