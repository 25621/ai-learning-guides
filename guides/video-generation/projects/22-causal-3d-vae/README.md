# Causal 3D VAE

## Key Insight

A plain [3D VAE](/shared/glossary/#3d-vae) compresses a fixed block of frames together, which means it cannot cleanly encode a *single* image — there is no neighboring frame to merge with. A [causal 3D VAE](/shared/glossary/#causal-3d-vae) fixes this by making each frame's encoding depend only on itself and *earlier* frames, never later ones — the same "look only backward" rule as a [causal mask](/shared/glossary/#causal-mask) in language models. The first frame is therefore encoded entirely on its own, so a one-frame input maps to a one-frame latent (`T=1 → T'=1`) and the very same model handles both still images and video. This is exactly why frontier video models use causal VAEs: they can [co-train on images and video](../16-joint-image-video-training/README.md) through one shared compressor instead of maintaining two.
