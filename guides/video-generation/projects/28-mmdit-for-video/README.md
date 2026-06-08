# MMDiT for Video

## Key Insight

A plain video [DiT](/shared/glossary/#dit) lets the video tokens *read* the text prompt through a one-way [cross-attention](/shared/glossary/#cross-attention) step — the text influences the video, but never the reverse. [MMDiT (Multi-Modal Diffusion Transformer)](/shared/glossary/#mmdit) — the [SD3](/shared/glossary/#sd3) and [Flux](/shared/glossary/#flux) design — instead sends text tokens and video tokens through the *same* [attention](/shared/glossary/#attention) layers, so the two [modalities](/shared/glossary/#modality) see and shape each other inside one shared operation. That two-way conversation is what helps the model get compositional prompts right ("a red cube *on* a blue sphere"), and building it for video lets you measure the payoff directly: text adherence should visibly improve over a cross-attention baseline.
