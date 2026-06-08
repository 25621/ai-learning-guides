# Aspect-Ratio Bucketing

## Key Insight

A batch of training clips must be a single [tensor](/shared/glossary/#tensor) of one [shape](/shared/glossary/#shape), so the lazy fix is to crop everything to a square — which slices the edges off tall phone videos and wide cinematic shots, teaching the model to frame only squares. [Aspect-ratio bucketing](/shared/glossary/#aspect-ratio-bucketing) instead sorts clips by their shape into a handful of buckets (tall, wide, square) and builds each [batch](/shared/glossary/#batch) from a single bucket, so nothing gets cropped and the model learns to generate at many shapes at once. This project implements bucketed batching and shows the quality jump on portrait and widescreen test prompts — the groundwork for [variable-resolution](/shared/glossary/#variable-resolution) inference, where one trained model can output any aspect ratio on demand.
