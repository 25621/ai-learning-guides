# GAN Inversion

## Key Insight

A trained GAN only runs one direction: noise in, image out. [GAN inversion](/shared/glossary/#gan-inversion) solves the reverse problem — given a real photo, find the [latent](/shared/glossary/#latent-space) code that makes the [generator](/shared/glossary/#generator) reproduce it. Once you have that code you can edit the real image by nudging it in latent space, which is why inversion is the bridge between "generate random faces" and "edit *this particular* face." This project finds the code two ways and compares the trade-off: first by directly optimizing the latent to minimize reconstruction error (slow but accurate), then by training an encoder that predicts it in a single forward pass (fast but approximate) — often inverting into the more expressive [W+ space](/shared/glossary/#w-and-w-latent-spaces) because its per-layer codes can match a real photo more closely.
