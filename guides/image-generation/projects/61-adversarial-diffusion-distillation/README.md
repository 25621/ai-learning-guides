# Adversarial Diffusion Distillation

## Key Insight

Pure few-step distillation tends to produce blurry images, because regressing toward an average washes out fine detail. [Adversarial Diffusion Distillation (ADD)](/shared/glossary/#add-adversarial-diffusion-distillation) — the recipe behind [SDXL Turbo](/shared/glossary/#sdxl-turbo) — fixes this by adding a [GAN](/shared/glossary/#gans)-style [discriminator](/shared/glossary/#discriminator) that rejects any quick output which doesn't look real, forcing the 1–4-step student to stay sharp. Comparing it head-to-head with an [LCM](/shared/glossary/#lcm) shows the trade-off plainly: ADD's adversarial training is fiddlier to run but yields crisper few-step samples.
