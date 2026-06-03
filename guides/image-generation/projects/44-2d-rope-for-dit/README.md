# 2D RoPE for DiT

## Key Insight

A [transformer](/shared/glossary/#transformer) has no built-in sense of position — to it, a sequence of image [patches](/shared/glossary/#patch) is just an unordered bag — so you must tell it *where* each patch sits. [2D RoPE (rotary position embedding)](/shared/glossary/#rope) does this by rotating each token's query and key vectors by an angle set from its row and column, so the [attention](/shared/glossary/#attention) [dot product](/shared/glossary/#dot-product) between two patches depends only on their relative spacing rather than their absolute coordinates. Swapping a [DiT](/shared/glossary/#dit)'s learned position vectors for 2D RoPE usually improves quality — especially when generating at resolutions larger than the model was trained on, because rotations extrapolate to unseen positions far more gracefully than a fixed lookup table of learned vectors.
