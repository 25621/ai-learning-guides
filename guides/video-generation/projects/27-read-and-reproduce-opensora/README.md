# Read and Reproduce OpenSora

## Key Insight

[OpenSora](/shared/glossary/#opensora) is a fully open replica of OpenAI's [Sora](/shared/glossary/#sora) recipe — a [DiT](/shared/glossary/#dit) running over [3D VAE](/shared/glossary/#3d-vae) latents and trained with [flow matching](/shared/glossary/#flow-matching) — and because its code, weights, and data pipeline are all public, it is the fastest way to see a complete Sora-style system end to end rather than from a paper's block diagram. Running inference on a pretrained checkpoint first builds intuition for what the model can and cannot do; then swapping out one component (for example, retraining with a different [VAE](/shared/glossary/#vae)) teaches which piece of the pipeline controls which failure mode. This is the project where the abstract architecture of [Phase 6](../../README.md#phase-6-diffusion-transformers-dit-and-sora-class-models) becomes a concrete thing you can run, break, and fix.
