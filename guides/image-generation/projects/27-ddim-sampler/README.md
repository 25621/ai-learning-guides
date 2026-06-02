# DDIM Sampler

## Key Insight

A plain [DDPM](/shared/glossary/#ddpm) samples by reversing its noising process one tiny *stochastic* step at a time, which can mean up to 1000 network calls to make a single image — accurate but painfully slow. [DDIM (Denoising Diffusion Implicit Models)](/shared/glossary/#ddim) rewrites that reverse process as a *deterministic* path: given the same starting noise it always lands on the same image, and because the path is smooth you can take big strides along it, skipping most of the steps. The headline result you will reproduce is that ~50 DDIM steps match the quality of a 1000-step DDPM — roughly a 20× speedup for almost no loss in quality — and the same trained model is reused unchanged, since DDIM only changes how you *sample*, not how you *train*.
