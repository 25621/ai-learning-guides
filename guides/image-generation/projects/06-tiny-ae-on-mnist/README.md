# Tiny AE on MNIST

## Key Insight

An [autoencoder](/shared/glossary/#autoencoder) is the simplest way to learn a compressed code for images. This project squeezes each 28×28 [MNIST](/shared/glossary/#mnist) digit (784 numbers) down to just 32 numbers and then rebuilds it, forcing the network to keep only what matters. Those 32 numbers form the [latent space](/shared/glossary/#latent-space), and the real magic is what happens *between* points: take the codes for a "3" and an "8", average them, decode the result, and you get a believable in-between digit. That smooth blending is the first sign the model learned genuine structure rather than memorizing pixels — and it is the foundation every fancier generator builds on. Notice there is no randomness yet: a plain autoencoder can rebuild and interpolate, but it cannot invent new digits from nothing, which is exactly the gap the VAE closes next.
