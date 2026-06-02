# Vanilla GAN on MNIST

## Key Insight

A [GAN](/shared/glossary/#gans) trains two networks in a contest: a [generator](/shared/glossary/#generator) that turns a vector of random noise into a fake image, and a [discriminator](/shared/glossary/#discriminator) that looks at an image and guesses whether it is real or generated. They improve by competing — the generator keeps trying to fool the critic while the critic keeps getting better at catching it — like a counterfeiter and a detective who each sharpen the other. This project builds [DCGAN](/shared/glossary/#dcgan), the first convolutional GAN recipe that trained reliably, on [MNIST](/shared/glossary/#mnist) digits, where you will watch the two losses oscillate instead of settling and learn to spot [mode collapse](/shared/glossary/#mode-collapse) — when the generator gives up on variety and keeps emitting the same one or two digits that happen to fool the critic.
