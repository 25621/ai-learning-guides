# Video Inversion and Edit

## Key Insight

To edit a *real* video with a [diffusion model](/shared/glossary/#diffusion-model) you first have to get it *into* the model's world, and that is what [DDIM inversion](/shared/glossary/#ddim-inversion) does: it runs the deterministic sampler backward to recover the exact starting noise that would regenerate the clip. Holding that noise, you change the text prompt — "a cat" → "a dog" — and denoise forward again (the [Prompt-to-Prompt](/shared/glossary/#prompt-to-prompt) style of edit), so the new object swaps in while the original's layout, motion, and timing are preserved. The video-specific catch is [temporal consistency](/shared/glossary/#temporal-consistency): inverting and editing each frame on its own makes the new object jitter between frames, so the edit has to be propagated coherently across time rather than recomputed independently per frame.
