# Implement ViT From Scratch

## Key Insight

A [ViT](/shared/glossary/#vit) treats an image as a short sentence whose "words" are square [patches](/shared/glossary/#patch): you cut the picture into a grid, project each block into a token ([patchification](/shared/glossary/#patchification)), prepend a learnable [CLS token](/shared/glossary/#cls-token), and run the sequence through ordinary [transformer](/shared/glossary/#transformer) blocks, then read the CLS token's final vector as the whole-image summary. Coding every piece yourself on [CIFAR-10](/shared/glossary/#cifar-10) — small enough to train on a laptop — makes the key abstraction concrete: the transformer never "sees" a 2D picture, only a list of vectors, so the same block that reads words reads pixels once you hand it patches.
