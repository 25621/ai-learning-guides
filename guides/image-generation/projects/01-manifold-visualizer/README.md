# Manifold Visualizer

## Key Insight

A 32×32 color image lives in a space with 3,072 numbers, but real photos do not fill that space — they cluster on a thin, curved surface called the [manifold](/shared/glossary/#manifold). This project draws 1,000 real CIFAR-10 images and 1,000 images of pure random noise, then squashes both groups down to 2D with [PCA](/shared/glossary/#pca-principal-component-analysis) so you can see them on a plot. The real images form a tight, structured blob while the random ones scatter everywhere, making the manifold visible to the eye. That single picture explains why generation is hard: the model must learn to land only on that thin blob and avoid the vast emptiness around it.
