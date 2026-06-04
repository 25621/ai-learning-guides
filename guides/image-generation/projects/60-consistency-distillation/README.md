# Consistency Distillation

## Key Insight

A 50-step [Stable Diffusion](/shared/glossary/#stable-diffusion) model is accurate but slow; [consistency distillation](/shared/glossary/#distillation) trains a [consistency model](/shared/glossary/#consistency-model) student that reaches a comparable image in just 4 steps by learning to jump straight to the denoising trajectory's endpoint instead of crawling along it. The practical, latent-space version is the [LCM](/shared/glossary/#lcm), which is what makes near-interactive image generation possible. The headline lesson is that [distillation](/shared/glossary/#distillation) moves cost from inference to training — you pay once to train the student so every future image is ~10× cheaper, accepting a modest quality dip in return.
