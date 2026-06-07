# Diffusion on Latents

## Key Insight

This project assembles the two halves of modern video generation: put the trained [3D VAE](/shared/glossary/#3d-vae) in front of a small video [diffusion model](/shared/glossary/#diffusion-model), so denoising runs entirely on compressed [latent video](/shared/glossary/#latent-video) instead of raw pixels — the same [latent-diffusion](/shared/glossary/#ldm) move that made [Stable Diffusion](/shared/glossary/#stable-diffusion) practical for images. Because the latent tensor is roughly 100× smaller, each training step is dramatically cheaper and far longer clips fit in memory than pixel-space diffusion could ever manage. Comparing the two side by side makes the headline result of Phase 5 concrete: latent diffusion trains faster *and* reaches higher quality at the same compute, because the model spends its capacity on motion and structure rather than on memorizing the pixel-level texture the VAE already reconstructs.
