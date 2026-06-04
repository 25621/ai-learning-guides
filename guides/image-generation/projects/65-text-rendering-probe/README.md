# Text-Rendering Probe

## Key Insight

[Text rendering](/shared/glossary/#text-rendering) — drawing legible, correctly-spelled words inside an image — was for years generation's most embarrassing failure, because a model that only matches overall image statistics learns the *shape* of letters but not that their exact order matters. This project builds a targeted probe: 200 prompts of the form "a sign that says '…'", run through open models, scoring how often the text comes out spelled right. Isolating one specific capability lets you rank models on it directly, and reveals how much newer models like [Flux](/shared/glossary/#flux) have closed a gap that older [Stable Diffusion](/shared/glossary/#stable-diffusion) models could not.
