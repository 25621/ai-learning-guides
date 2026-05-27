# Mini-MoE

---

> Keep many [expert](/shared/glossary/#expert) networks on hand, but pay to run only the few each token actually needs.

---

## Key Insight

A [Mixture-of-Experts (MoE)](/shared/glossary/#moe) replaces one [MLP](/shared/glossary/#mlp) with several "expert" MLPs plus a router that sends each token to only the top few experts. Total parameters grow large while the compute spent per token stays fixed.

## Why This Matters

MoE is how models like Mixtral and DeepSeek-V3 reach huge parameter counts affordably. Adding an 8-expert top-2 layer to nanoGPT — and watching whether the router spreads tokens evenly across experts — exposes the central challenge of MoE: keeping routing balanced.
