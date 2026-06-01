# Hierarchical VAE

## Key Insight

A [hierarchical VAE](/shared/glossary/#hierarchical-vae) stacks more than one layer of latent variables at different sizes — here an 8×8 grid and a smaller 4×4 grid — so the model can split its work across scales. The coarse 4×4 level captures the big picture (overall shape and color) while the finer 8×8 level fills in local detail, much like sketching rough shapes before adding texture. This project trains such a two-level model on [CIFAR-10](/shared/glossary/#cifar-10) and asks a simple question: does dividing the labor across scales beat a single flat [latent space](/shared/glossary/#latent-space)? For natural images, which contain structure at many sizes at once, the answer is usually yes — and that is why the strongest VAEs and modern image generators are all multi-scale.
