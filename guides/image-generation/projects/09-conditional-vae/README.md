# Conditional VAE

## Key Insight

A plain [VAE](/shared/glossary/#vae) can dream up new [MNIST](/shared/glossary/#mnist) digits, but it picks *which* digit at random — you cannot ask it for a "7." [Class conditioning](/shared/glossary/#class-conditioning) fixes this by feeding the digit label into both the encoder and the decoder, so the model learns a separate region of its latent space for each class. At generation time you simply hand it the label you want, draw a random latent, and reliably get that exact digit. This tiny change is the seed of all controllable generation: the same idea, scaled up, is how text-to-image models turn a written prompt into the picture you asked for.
