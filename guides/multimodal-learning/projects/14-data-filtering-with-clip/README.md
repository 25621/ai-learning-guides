# Data Filtering with CLIP

## Key Insight

The same [cosine similarity](/shared/glossary/#cosine-similarity) that [CLIP](/shared/glossary/#clip) computes between an image and its caption — the *CLIP score* — doubles as a cheap quality detector for noisy web image–text data: a low score usually means the alt-text is keyword spam or simply unrelated to the picture, so dropping the bottom-scoring pairs throws out the noise that would otherwise confuse a model. Training one [downstream](/shared/glossary/#downstream) model on the filtered set and another on the raw set typically shows the filtered model winning *despite* seeing fewer examples — proof that for web data, quality beats raw quantity. This is exactly the filter that was used to build [LAION](/shared/glossary/#laion) and that opens nearly every large image–text data-curation pipeline.
