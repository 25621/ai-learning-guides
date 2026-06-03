# Train a Latent DDPM

## Key Insight

This project is a miniature of how [Stable Diffusion](/shared/glossary/#stable-diffusion) actually works: instead of running [DDPM](/shared/glossary/#ddpm) directly on pixels, you first compress each [CIFAR-10](/shared/glossary/#cifar-10) image with an 8× [VAE](/shared/glossary/#vae) into a tiny 4×4 [latent](/shared/glossary/#latent-space), then train a small [U-Net](/shared/glossary/#u-net) to denoise in that latent grid — the core idea of [latent diffusion (LDM)](/shared/glossary/#ldm). Because the latent has roughly 48× fewer numbers than the pixels, every training and sampling step is dramatically cheaper, and you decode back to pixels only at the very end. Comparing it to a pixel-space DDPM on the same data makes the central lesson concrete: most of diffusion's cost is spent on pixel detail the VAE can reconstruct anyway.
