# Textual Inversion

## Key Insight

[Textual Inversion](/shared/glossary/#textual-inversion) takes the opposite tack from [DreamBooth](/shared/glossary/#dreambooth): it changes *nothing* in the [diffusion model](/shared/glossary/#diffusion-model) itself and instead learns a single new [word embedding](/shared/glossary/#embedding) — a fresh row added to the [text encoder](/shared/glossary/#clip)'s [embedding matrix](/shared/glossary/#embedding-matrix) — that points at your subject. Because only that one vector is trained, the result is a few kilobytes, the smallest personalization artifact there is. The catch is capacity: a single vector can capture a recognizable "vibe" but cannot match the fidelity of [LoRA](/shared/glossary/#lora) or DreamBooth, because the frozen [weights](/shared/glossary/#weights) can only render what the model already knows how to draw.
