# Mini LAION Pipeline

## Key Insight

A modern [text-to-image](/shared/glossary/#text-to-image) model is only as good as the data it eats, and raw web scrapes like [LAION](/shared/glossary/#laion) are mostly noise — duplicates, mismatched captions, and junk images. This project walks the real production pipeline on a small shard: download the image URLs, drop near-duplicates with [deduplication](/shared/glossary/#deduplication), keep only images whose caption actually matches the picture using a [CLIP](/shared/glossary/#clip) similarity score, filter for visual appeal with an [aesthetic score](/shared/glossary/#aesthetic-score), and rewrite weak alt-text into rich descriptions using [synthetic captions](/shared/glossary/#synthetic-captions) from a small [VLM](/shared/glossary/#vlm). The lesson is that data engineering — not architecture — is where most of a generator's quality is won or lost.
