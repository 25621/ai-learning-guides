# DDIM Inversion and Editing

## Key Insight

To edit a *real* photo with a [diffusion model](/shared/glossary/#diffusion-model) you first need the noise that would regenerate it — and [DDIM inversion](/shared/glossary/#ddim-inversion) finds it by running the deterministic [DDIM](/shared/glossary/#ddim) sampler *backwards*, adding noise along the same path the model would later remove. Once you hold that starting noise you change the prompt and denoise forward again: because the trajectory is largely shared, the new image keeps the original's layout and pose while swapping the content you re-described. The imperfection you will see — inversion drifts, so the reconstruction is never pixel-perfect — is exactly the gap that follow-up methods like *null-text inversion* were built to close.
