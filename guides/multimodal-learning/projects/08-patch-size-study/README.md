# Patch-Size Study

## Key Insight

A [ViT](/shared/glossary/#vit)'s [patch](/shared/glossary/#patch) size is the single knob that trades accuracy for compute. Halving the patch edge (32 → 16 → 8) quarters the area each token covers, so the token count rises — and because [attention](/shared/glossary/#attention) cost grows with the square of the sequence length, the [FLOPs](/shared/glossary/#flops) climb steeply, buying the fine detail the model needs for small objects and text. Running one ViT at patch 8, 16, and 32 and plotting accuracy against FLOPs lets you *see* that curve and pick the smallest patch you can afford — exactly the tradeoff the "/16" or "/8" suffix in a name like ViT-B/16 is quietly announcing.
