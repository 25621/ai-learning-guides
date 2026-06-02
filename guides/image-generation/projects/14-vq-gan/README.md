# VQ-GAN

## Key Insight

A plain [VQ-VAE](/shared/glossary/#vq-vae) trained only to match pixels produces blurry reconstructions, because averaging over many plausible details is the safest way to lower a pixel-by-pixel error. [VQ-GAN](/shared/glossary/#vq-gan) fixes this by adding two extra training signals: a [perceptual loss](/shared/glossary/#perceptual-loss-lpips) that compares images in the feature space of a pretrained network (so it cares about textures and shapes, not exact pixels), and a patch discriminator — a small critic borrowed from [GANs](/shared/glossary/#gans) that judges whether each local region of the image looks real or fake. Together they push the decoder to commit to sharp, specific details instead of hedging. This project adds both to your VQ-VAE and watches reconstructions go from soft to crisp — the same recipe that [Stable Diffusion](/shared/glossary/#stable-diffusion)'s VAE descends from.
