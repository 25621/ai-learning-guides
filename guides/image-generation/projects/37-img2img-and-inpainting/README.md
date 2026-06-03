# img2img and Inpainting

## Key Insight

[img2img](/shared/glossary/#img2img) and [inpainting](/shared/glossary/#inpainting) show that the same trained [diffusion model](/shared/glossary/#diffusion-model) does far more than generate from scratch — both fall out of the [Stable Diffusion](/shared/glossary/#stable-diffusion) sampling loop almost for free. The key control is the *noise strength* (also called denoising strength): instead of starting from pure noise, you partially noise an existing image and denoise from there, so a low strength keeps the original almost intact while a high strength redraws it freely. Inpainting adds a mask so only the selected region is regenerated while the known pixels are pasted back on every step — exactly how you erase an object or swap a face without disturbing the rest of the picture.
