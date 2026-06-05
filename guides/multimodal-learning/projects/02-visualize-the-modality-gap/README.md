# Visualize the Modality Gap

## Key Insight

[CLIP](/shared/glossary/#clip) is trained to drop an image and its caption onto the *same* spot in a shared space, so you might expect the picture of a dog and the words "a dog" to land right on top of each other. They do not. If you encode 1,000 images and 1,000 captions and squash all 2,000 [embeddings](/shared/glossary/#embedding) down to a 2D plot with [PCA](/shared/glossary/#pca-principal-component-analysis), you will see the [modality gap](/shared/glossary/#modality-gap). Seeing the gap with your own eyes explains a string of later surprises: why [cosine similarity](/shared/glossary/#cosine-similarity) scores between correctly matched image–text pairs are lower than you would guess, and why some retrieval and generation methods add a correction step that shifts one modality toward the other to close it.
