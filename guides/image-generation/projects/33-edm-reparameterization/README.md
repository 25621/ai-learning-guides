# EDM Reparameterization

## Key Insight

[EDM](/shared/glossary/#edm) (Karras et al. 2022) doesn't change what a [diffusion model](/shared/glossary/#diffusion-model) *is* — it changes the bookkeeping so training and sampling stop fighting you. Instead of indexing noise by a discrete timestep `t`, it uses the noise standard deviation σ directly (the [σ-schedule](/shared/glossary/#σ-schedule-karras)), and it adds *preconditioning*: the network's input, output, and per-σ loss weight are each rescaled so the network always sees roughly unit-variance signals no matter how much noise is present. Without this, the network wastes capacity learning to rescale wildly different magnitudes across noise levels; with it, the hyperparameter surface becomes flat and forgiving, so good results stop depending on lucky tuning. This project re-derives your [CIFAR-10](/shared/glossary/#cifar-10) model in σ-space and lets you observe the cleaner training and sampling behavior firsthand.
