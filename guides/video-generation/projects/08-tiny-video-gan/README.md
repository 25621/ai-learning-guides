# Tiny Video GAN

## Key Insight

This project trains a small [video GAN](/shared/glossary/#video-gan) on face crops from [UCF-101](/shared/glossary/#ucf-101) so you can watch [mode collapse](/shared/glossary/#mode-collapse) happen with your own eyes — the [generator](/shared/glossary/#generator) discovers a couple of clips that reliably fool the [discriminator](/shared/glossary/#discriminator) and then keeps producing only those, ignoring the variety in the real data. [GANs](/shared/glossary/#gans) extend awkwardly to video because the discriminator must judge not just whether each frame looks real but whether the *motion* between frames does, which makes the adversarial game even less stable than it is for images. Living through that instability is the point: it is the concrete reason the field abandoned GANs for [diffusion](/shared/glossary/#diffusion-model) once diffusion proved both sharper and far more stable to train.
