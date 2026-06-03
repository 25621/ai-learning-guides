# Compare DiT and U-Net Scaling

## Key Insight

The reason the field switched from [U-Nets](/shared/glossary/#u-net) to [Diffusion Transformers (DiT)](/shared/glossary/#dit) is not that a small DiT beats a small U-Net — often it does not — but that DiT obeys cleaner [scaling laws](/shared/glossary/#scaling-laws): as you grow the model and the compute you spend on it, quality improves along a smoother, steeper line. Training DiT-S, DiT-B, and DiT-L on the same data and plotting [FID](/shared/glossary/#fid) against [FLOPs](/shared/glossary/#flops) makes that slope visible — the DiT curve keeps dropping where the U-Net flattens out. This is the same lesson that played out in language modeling: predictable scaling beats clever architecture once you can afford to scale.
