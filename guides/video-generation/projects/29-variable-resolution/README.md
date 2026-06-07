# Variable Resolution

## Key Insight

Most diffusion models are welded to the one resolution they trained on; [Sora](/shared/glossary/#sora)'s headline claim was [variable resolution](/shared/glossary/#variable-resolution) — one model that generates clips at many sizes, durations, and aspect ratios. It works because a [DiT](/shared/glossary/#dit) processes a *sequence* of [spatiotemporal patches](/shared/glossary/#spatiotemporal-patches) rather than a fixed grid, and [3D RoPE](/shared/glossary/#rope) encodes each token's position as a rotation that extrapolates gracefully to lengths and shapes never seen in training. By feeding the model more or fewer tokens you get a taller, wider, or longer video from the very same weights. Training across [aspect-ratio buckets](/shared/glossary/#aspect-ratio-bucketing) and then testing on an aspect ratio it never saw is the concrete experiment that proves the claim — or exposes where it breaks down.
