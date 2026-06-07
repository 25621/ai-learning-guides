# Flow Matching from Scratch

## Key Insight

[DDPM](/shared/glossary/#ddpm) trains a model to undo noise one small, carefully scheduled step at a time; [flow matching](/shared/glossary/#flow-matching) throws away the schedule and trains a *velocity field* instead — a model that, given a half-noisy clip and a time, predicts the single straight-line direction pointing back toward clean video. [Rectified flow](/shared/glossary/#rectified-flow) is the variant that makes those paths actually straight, which is why generation needs far fewer sampling steps than the curvy trajectories of older diffusion. Swapping DDPM for rectified flow inside a small video [DiT](/shared/glossary/#dit) and watching it converge faster makes concrete why nearly every 2024+ model — [SD3](/shared/glossary/#sd3), [Flux](/shared/glossary/#flux), and most frontier video models — trains this way instead.
