# Read MoCoGAN

## Key Insight

[MoCoGAN](/shared/glossary/#mocogan)'s lasting idea is to split a video's [latent](/shared/glossary/#latent-space) code into two pieces — a single *content* vector that stays fixed across the whole clip (who or what is in it) and a sequence of *motion* vectors that change each frame (how it moves). This project implements just that decomposition inside a small [VAE](/shared/glossary/#vae): hold content steady and vary motion, and the same subject performs different movements without morphing into someone else partway through. The reason it is worth studying a 2017 model in a diffusion world is that this exact content/motion separation keeps reappearing — as keyframe-plus-motion conditioning, as reference-image identity locks — inside modern systems. The framework was wrong; the decomposition was right.
