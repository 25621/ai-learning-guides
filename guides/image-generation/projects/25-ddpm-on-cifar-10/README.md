# DDPM on CIFAR-10

## Key Insight

Moving a [DDPM (Denoising Diffusion Probabilistic Model)](/shared/glossary/#ddpm) from grayscale [MNIST](/shared/glossary/#mnist) digits to 32×32 color [CIFAR-10](/shared/glossary/#cifar-10) photos is the jump from "it works on a toy" to "I actually understand this," because natural images carry texture, color, and structure that a too-small [U-Net](/shared/glossary/#u-net) simply cannot capture. The standard bar is a [FID (Fréchet Inception Distance)](/shared/glossary/#fid) below 20 — FID scores how close your generated images are to real ones by comparing them in the feature space of a pretrained image classifier, where lower means more realistic — and clearing it forces you to get the [noise schedule](/shared/glossary/#noise-schedule), model capacity, and training length all right at once. The payoff is a model whose samples are recognizable objects rather than colorful blobs, plus the confidence that the core recipe scales beyond toy data.
