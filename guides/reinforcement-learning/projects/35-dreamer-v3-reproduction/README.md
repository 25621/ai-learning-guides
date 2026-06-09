# Dreamer V3 Reproduction

## Key Insight

[DreamerV3](/shared/glossary/#dreamerv3) learns a compact [world model](/shared/glossary/#world-model) of the environment and then trains its [policy](/shared/glossary/#policy) almost entirely "in imagination" — rolling the policy forward inside the learned [latent dynamics](/shared/glossary/#latent-dynamics) instead of the slow, expensive real environment. Its claim to fame is generality: a single fixed set of hyperparameters clears tasks as different as [Atari](/shared/glossary/#atari) games and collecting diamonds in Minecraft, which had long been an open problem in RL where each task usually needs its own tuning. Reproducing it on a custom environment is the best way to feel how much work the [model-based](/shared/glossary/#model-based-rl) world model does and how few knobs you actually have to touch.
