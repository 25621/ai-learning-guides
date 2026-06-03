# ControlNet from Scratch

## Key Insight

[ControlNet](/shared/glossary/#controlnet) gives a [diffusion model](/shared/glossary/#diffusion-model) precise spatial control by cloning the [U-Net](/shared/glossary/#u-net)'s encoder into a parallel branch that reads a conditioning image — here a [Canny edge map](/shared/glossary/#canny-edge-detector) — and injects its features back into the frozen original. The trick that makes this trainable without wrecking the pretrained model is the [zero-convolution](/shared/glossary/#zero-conv): the connections start at exactly zero, so on step one the branch contributes nothing and the base model behaves as before, then the zero-convs gradually learn how much control to add. Building it from scratch on Canny edges makes the core idea concrete — the model learns to *trace* the edge map while still inventing texture, color, and lighting from the prompt.
