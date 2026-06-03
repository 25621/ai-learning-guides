# Train a VAE for Diffusion

## Key Insight

Every [latent diffusion](/shared/glossary/#ldm) model is only as good as the [VAE](/shared/glossary/#vae) it generates in, so this project trains that compressor properly before any diffusion happens — on [CelebA](/shared/glossary/#celeba) faces, where it is easy to judge whether reconstructions look real. The recipe is the one [Stable Diffusion](/shared/glossary/#stable-diffusion)'s VAE descends from: combine a [perceptual loss (LPIPS)](/shared/glossary/#perceptual-loss-lpips) for sharp textures, an adversarial loss from a [discriminator](/shared/glossary/#discriminator) so fine detail looks real instead of blurry, and a light [KL](/shared/glossary/#kl-divergence) penalty to keep the latent space smooth enough to diffuse in. The point is to verify the VAE is a faithful compressor first — a leaky one silently caps the quality of any diffusion model you later train on its latents.
