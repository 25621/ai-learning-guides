# Reparameterization Audit

## Key Insight

[SAC](/shared/glossary/#sac)'s actor outputs a Gaussian that is then squeezed through a [tanh squashing](/shared/glossary/#tanh-squashing) function so every action stays inside the environment's bounds, and it trains that actor with the [reparameterization trick](/shared/glossary/#reparameterization-trick) — sampling plain noise and pushing it through `mean + std · ε` so [gradients](/shared/glossary/#gradients) flow through the otherwise-random action. The catch is that squashing a Gaussian through `tanh` changes its probability density, so the log-probability used in the loss needs a correction term; get that correction wrong and SAC keeps running while silently learning the wrong [entropy](/shared/glossary/#entropy-regularization), producing a policy that looks trained but quietly underperforms. Auditing this one formula by hand is the cheapest way to catch a bug that no error message will ever report.
