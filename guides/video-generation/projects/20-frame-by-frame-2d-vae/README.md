# Frame-by-Frame 2D VAE

## Key Insight

The cheapest way to push video into a [latent space](/shared/glossary/#latent-space) is to run a [Stable Diffusion](/shared/glossary/#stable-diffusion) image [VAE](/shared/glossary/#vae) on each frame on its own — no new training, no time axis. But because the VAE never sees more than one frame at a time, it makes tiny independent rounding errors that differ from frame to frame, and those differences surface as [temporal flicker](/shared/glossary/#temporal-flicker): textures shimmer and flat areas pulse even when the real scene is perfectly still. This project makes that failure visible and motivates the [3D VAE](/shared/glossary/#3d-vae) in the next project, which compresses across time too and so keeps reconstructions stable. The lesson is that per-frame compression throws away the one thing video has that a stack of images doesn't — the fact that neighboring frames are almost identical.
