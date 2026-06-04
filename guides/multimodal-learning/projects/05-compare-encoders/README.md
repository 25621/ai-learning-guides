# Compare Encoders

## Key Insight

The fairest way to ask "which encoder sees the world best" is to freeze each one, pull its [features](/shared/glossary/#embedding) for the same set of [ImageNet](/shared/glossary/#imagenet) images, and fit a [linear probe](/shared/glossary/#linear-probe) on top: if a single linear layer can separate the classes, the encoder already did the hard work. Comparing a convolutional [ResNet](/shared/glossary/#resnet)-50 against a [ViT](/shared/glossary/#vit), a contrastively trained [SigLIP](/shared/glossary/#siglip), and a label-free [self-supervised](/shared/glossary/#self-supervised) [DINOv2](/shared/glossary/#dinov2) on one probe reveals that *how* a model was trained often matters more than its architecture — DINOv2, which never saw a label, frequently beats supervised towers, which is why it has become a default off-the-shelf vision backbone in 2026.
