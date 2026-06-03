# MMDiT Block

## Key Insight

In a standard [DiT](/shared/glossary/#dit), text guides the image through [cross-attention](/shared/glossary/#cross-attention) — the image tokens look at the text tokens, but not the other way around. [MMDiT (Multi-Modal Diffusion Transformer)](/shared/glossary/#mmdit), the block behind [Stable Diffusion 3](/shared/glossary/#stable-diffusion) and [Flux](/shared/glossary/#flux), instead concatenates text and image tokens and runs them through *one shared* [self-attention](/shared/glossary/#attention) ("joint attention"), while giving each [modality](/shared/glossary/#modality) its own normalization and [MLP](/shared/glossary/#mlp) weights. Letting text and image attend to each other freely in both directions improves compositional prompts — getting "a red cube on a blue sphere" right instead of swapping the colors — which is why most strong 2024+ open models adopted it.
