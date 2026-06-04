# Caption Ablation

## Key Insight

This is a controlled [ablation](/shared/glossary/#ablation): train two otherwise-identical small [text-to-image](/shared/glossary/#stable-diffusion) models that differ in exactly one thing — one sees the original web alt-text, the other sees [synthetic captions](/shared/glossary/#synthetic-captions) rewritten by a [VLM](/shared/glossary/#vlm) — so any quality gap is caused by the captions alone. The recaptioned model will follow prompts noticeably better, which is the open-source confirmation of the trick behind [DALL·E 3](/shared/glossary/#synthetic-captions)'s compositional skill. It teaches the most counter-intuitive lesson in the field: improving the *captions* often beats improving the *model*.
