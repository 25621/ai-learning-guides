# Conditional GAN

## Key Insight

A plain GAN samples a random image from everything it learned, with no way to ask for a specific kind. A [conditional GAN (cGAN)](/shared/glossary/#conditional-gan-cgan) fixes this by feeding the class label to *both* the [generator](/shared/glossary/#generator) and the [discriminator](/shared/glossary/#discriminator), turning generation into [class conditioning](/shared/glossary/#class-conditioning) — you can request "a 7" or "a cat" on demand. This project injects the label through a [projection discriminator](/shared/glossary/#projection-discriminator), which blends the label into the critic's score with a dot product against a learned per-class vector instead of just stapling the label on as an extra input channel — a trick that conditions more strongly for little extra cost. Training on [CIFAR-10](/shared/glossary/#cifar-10) with labels lets you verify the control works: ask for a class, and that class is what comes out.
