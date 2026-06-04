# Watermarking

## Key Insight

As AI images flood the internet, being able to prove an image was machine-generated becomes a safety requirement, and [watermarking](/shared/glossary/#watermarking) embeds an invisible, detectable signal that says "made by AI" without changing how the picture looks. In this project you add such a mark to your generator's outputs — either stamped into the pixels afterward or baked into the sampling process like [SynthID](/shared/glossary/#synthid) — then build a detector and confirm it flags your images while leaving real photos alone. The core tension you will feel is robustness vs invisibility: a mark strong enough to survive cropping and JPEG compression is harder to keep imperceptible.
