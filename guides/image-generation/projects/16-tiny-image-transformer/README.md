# Tiny Image Transformer

## Key Insight

Once a [VQ-GAN](/shared/glossary/#vq-gan) has turned an image into a grid of discrete [tokens](/shared/glossary/#token-visualaudio), a picture becomes "just another sentence" — a sequence of symbols from a fixed vocabulary — and the same machinery used for language applies directly. This project trains a small [transformer](/shared/glossary/#transformer) to predict those image tokens one after another in raster (row-by-row) order, exactly how an [autoregressive](/shared/glossary/#autoregressive-model) language model writes text word by word. To generate a new image you sample tokens one at a time and then decode the finished grid back into pixels with the VQ-GAN decoder. It is slow because each token must wait for the previous one, but it shows clearly why "image generation as language modeling" is such a powerful idea.
