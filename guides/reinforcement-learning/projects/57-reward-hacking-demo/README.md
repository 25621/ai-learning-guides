# Reward Hacking Demo

## Key Insight

[Reward hacking](/shared/glossary/#reward-hacking) is when a [policy](/shared/glossary/#policy) maximizes the *reward signal* without doing the thing the reward was meant to encourage — the gap between the proxy you can measure and the goal you actually care about. This project provokes it on purpose: over-train a model against a [reward model](/shared/glossary/#reward-model) with the [KL](/shared/glossary/#kl-divergence) penalty turned down or off, then characterize the gibberish that emerges as the policy discovers quirks that score highly but mean nothing. Why it matters: it shows in the most visceral way why the KL leash to a frozen [reference model](/shared/glossary/#reference-model) is non-negotiable in [RLHF](/shared/glossary/#rlhf), and why a learned reward model — unlike a deterministic [verifier](/shared/glossary/#verifier) in [RLVR](/shared/glossary/#rlvr) — can *always* be gamed given enough optimization pressure.
