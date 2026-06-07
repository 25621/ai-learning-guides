# Tiny I2V Model

## Key Insight

This is the smallest honest version of building your own [image-to-video (I2V)](/shared/glossary/#i2v) model: start from a [frozen](/shared/glossary/#frozen) [Stable Diffusion 1.5](/shared/glossary/#stable-diffusion) [U-Net](/shared/glossary/#u-net), insert new [(2+1)D](/shared/glossary/#21d) temporal convolution layers, and [fine-tune](/shared/glossary/#fine-tuning) only those new layers on ~100k clips while the first frame is fed in as the condition. Because *any* video is automatically a training example — its first frame is the input and the remaining frames are the target — you need no paired text captions at all, which is why I2V is the cheapest place to start training. Freezing the image backbone and training only the temporal layers is [temporal inflation](/shared/glossary/#temporal-inflation) in its rawest form: you keep everything the image model already knows about appearance and teach it only how those pixels should move over time.
