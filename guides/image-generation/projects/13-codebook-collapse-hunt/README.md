# Codebook Collapse Hunt

## Key Insight

With a discrete [codebook](/shared/glossary/#codebook), a common failure is [codebook collapse](/shared/glossary/#codebook-collapse): the model leans on just a handful of entries and lets the rest go unused, like a writer who knows ten thousand words but only ever reaches for the same fifty. A collapsed codebook wastes capacity and caps how much detail the model can store, so reconstructions stay blurry no matter how long you train. This project deliberately provokes the problem by using a codebook far larger than needed, then fixes it with two standard tools: [EMA](/shared/glossary/#ema-weights) updates that smoothly nudge entries toward the data they should represent, and dead-code re-initialization that recycles never-used entries by moving them next to popular ones. Watching the usage histogram fill in is the clearest way to feel what "collapse" really means.
