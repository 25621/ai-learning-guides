# Pre-Norm vs Post-Norm

---

> Where you place the normalization decides whether training is smooth or blows up.

---

## Key Insight

Pre-norm puts the normalization step *inside* each [residual](/shared/glossary/#residual-connection) branch (`x = x + Attn(Norm(x))`), while post-norm normalizes *after* the residual is added. Pre-norm trains stably even without learning-rate [warmup](/shared/glossary/#warmup); post-norm often needs warmup and can diverge without it.

## Why This Matters

Every modern [transformer](/shared/glossary/#transformer) is pre-norm for exactly this reason. Training two otherwise-identical models, with and without warmup, turns an abstract design rule into something you have watched succeed and fail with your own eyes.
