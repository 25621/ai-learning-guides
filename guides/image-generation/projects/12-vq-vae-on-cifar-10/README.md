# VQ-VAE on CIFAR-10

## Key Insight

A [VQ-VAE](/shared/glossary/#vq-vae) is an [autoencoder](/shared/glossary/#autoencoder) whose hidden code is forced to be *discrete*: instead of any continuous numbers, the encoder must describe each patch of the image using entries chosen from a small fixed list called a [codebook](/shared/glossary/#codebook) — like painting only with the colors in a numbered paint set. This project builds one on [CIFAR-10](/shared/glossary/#cifar-10) with a 256-entry codebook that turns each image into an 8×8 grid of code indices, then decodes that grid back into pixels. By plotting how often each codebook entry is used and comparing the rebuilt images to the originals, you can see how a tiny vocabulary of learned patterns is enough to reconstruct whole pictures. The trick that makes training possible — passing gradients straight through the non-differentiable lookup — is the [straight-through estimator](/shared/glossary/#straight-through-estimator).
