# Temperature Ablation

## Key Insight

The [temperature](/shared/glossary/#temperature) τ divides the [cosine-similarity](/shared/glossary/#cosine-similarity) scores before the [softmax](/shared/glossary/#softmax) in [InfoNCE](/shared/glossary/#infonce), tuning how sharp or soft that contest between the true pair and its [hardest negatives](/shared/glossary/#hard-negatives) becomes — and this project sweeps it end to end instead of trusting the default. Running τ from 0.01 to 1.0 and watching both accuracy and the [geometry of the embedding space](/shared/glossary/#embedding-space) — how tightly true pairs cluster on the unit sphere — makes visible why [CLIP](/shared/glossary/#clip) makes τ a *learned* parameter (initialized around 0.07) rather than a guessed constant. Set it too low and training is unstable and overconfident; too high and the model never commits — which is why this one scalar punches far above its weight.
