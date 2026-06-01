# β-VAE Study

## Key Insight

A [β-VAE](/shared/glossary/#β-vae) adds a single knob, β, that multiplies the [KL divergence](/shared/glossary/#kl-divergence) term of the VAE's loss — the part that pushes the latent code to stay tidy and close to a standard bell curve. This project sweeps β from 0 up to 10 and watches the trade-off: low β reconstructs sharply but leaves a messy latent space, while high β organizes the latent beautifully yet blurs the output. Push β too high and you trigger [posterior collapse](/shared/glossary/#posterior-collapse), where the decoder learns to do the job alone and ignores the latent entirely, so the code stops carrying any information about the input. Seeing both failure directions in one sweep teaches you that generative training is always a balancing act, never a single "best" setting.
