# Latent Traversal

## Key Insight

Once a [VAE](/shared/glossary/#vae) is trained on a dataset of faces like [CelebA](/shared/glossary/#celeba), its [latent space](/shared/glossary/#latent-space) is not just random storage — individual directions in it often line up with human-meaningful features. A *traversal* means freezing every latent number except one, then slowly turning that single dial up and down and decoding at each step to watch the face change. Do this across many dimensions and you discover that one controls hair color, another a smile, another the lighting direction — without anyone ever labeling those concepts during training. This is the clearest hands-on proof that a good generative model does not memorize images; it discovers the hidden knobs that the data varies along.
