# Causal 3D VAE

## Key Insight

A [causal 3D VAE](/shared/glossary/#causal-3d-vae) is a video compressor built so that each frame is encoded using only itself and earlier frames. This allows it to cleanly encode a *single* isolated image without needing later frames to merge with. This is exactly why frontier video models use them: they can [co-train on images and video](../16-joint-image-video-training/README.md) through one shared compressor instead of maintaining two separate encoders.
