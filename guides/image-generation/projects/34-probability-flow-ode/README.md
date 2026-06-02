# Probability Flow ODE

## Key Insight

Every [diffusion model](/shared/glossary/#diffusion-model) can be sampled two equivalent ways: as a noisy stochastic process — an [SDE](/shared/glossary/#sde-stochastic-differential-equation) that injects fresh randomness at every step — or as a single deterministic trajectory, the [probability flow ODE](/shared/glossary/#probability-flow-ode), which shares the *exact same* distribution at every noise level. Determinism buys two things the stochastic sampler can't: the same starting noise always maps to the same image (so you can smoothly interpolate between samples and invert a real image back to its noise), and because an ODE has a well-defined change-of-variables, you can compute the model's exact log-likelihood — how probable it thinks any given image is. This project converts an SDE sampler to its ODE form, checks the samples still look right, and computes those exact likelihoods.
