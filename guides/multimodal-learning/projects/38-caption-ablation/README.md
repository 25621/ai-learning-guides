# Caption Ablation

## Key Insight

An [ablation](/shared/glossary/#ablation) isolates the effect of one variable by changing only that variable and holding everything else fixed; here the variable is the *captions*. You train two otherwise-identical small [VLMs](/shared/glossary/#vlm) — one on the original web [alt-text](/shared/glossary/#alt-text), one on [synthetic captions](/shared/glossary/#synthetic-captions) rewritten by a stronger model — over the same images, then compare their [downstream](/shared/glossary/#downstream) scores. The gap is reliably large and in the same direction: detailed, accurate captions teach the model which words map to which pixels, so the recaptioned model wins by a wide margin even though it saw the exact same pictures. This is the cleanest way to *feel* the central claim of training at scale — that in multimodal learning, caption quality, not model size, is usually the dominant lever.
