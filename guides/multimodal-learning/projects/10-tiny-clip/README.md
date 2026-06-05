# Tiny CLIP

## Key Insight

Building a small [CLIP](/shared/glossary/#clip) end to end — a little [ViT](/shared/glossary/#vit) for images, a small transformer [text encoder](/shared/glossary/#text-encoder), and the [InfoNCE](/shared/glossary/#infonce) loss tying them together — shows how little the architecture matters and how much the training setup does. The single biggest lever is [batch size](/shared/glossary/#batch): every other caption in the batch acts as a negative example, so a batch of 1,024 gives each image a thousand wrong answers to be pushed away from, while a batch of 32 gives it only thirty-one and learns a far blurrier shared space. Train it on a modest set of image–caption pairs and you reproduce CLIP's headline trick in miniature — [zero-shot](/shared/glossary/#zero-shot) matching of unseen captions to images — while feeling firsthand why the original paper fought so hard for huge batches.
