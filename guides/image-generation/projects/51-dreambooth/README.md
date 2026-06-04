# DreamBooth

## Key Insight

[DreamBooth](/shared/glossary/#dreambooth) personalizes a [diffusion model](/shared/glossary/#diffusion-model) by [fine-tuning](/shared/glossary/#fine-tuning) *the whole network* on just 3–5 photos of a subject, binding it to a rare trigger word so you can later prompt "a photo of [V] dog on the moon." Because every [weight](/shared/glossary/#weights) is updated the likeness is excellent, but the saved file is the size of the full model — the opposite trade-off from a lightweight [LoRA](/shared/glossary/#lora). The danger is [catastrophic forgetting](/shared/glossary/#catastrophic-forgetting): train so hard on five images that the model forgets what every *other* dog looks like, which is exactly why DreamBooth adds a [prior-preservation loss](/shared/glossary/#prior-preservation-loss) — extra training on the model's own generic class images so the broad concept survives while the specific subject is learned.
