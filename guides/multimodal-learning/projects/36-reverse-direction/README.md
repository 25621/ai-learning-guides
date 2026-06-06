# Reverse Direction

## Key Insight

A standard [VLM](/shared/glossary/#vlm) only runs one way — pixels in, text out — because its output layer can emit text tokens and nothing else. Adding an image-token output head (a second prediction layer that emits [discrete image tokens](/shared/glossary/#token-visualaudio) drawn from a [VQ-VAE](/shared/glossary/#vq-vae) [codebook](/shared/glossary/#codebook)) lets the same model predict picture codes as easily as words, which a decoder then turns back into pixels — converting an understanding-only model into one that can generate images too. This is the cheapest path toward any-to-any: instead of retraining a [native multimodal](/shared/glossary/#native-multimodal) model from scratch, you graft an output head onto a model that already "speaks image" on the input side and teach it to speak image on the output side as well.
