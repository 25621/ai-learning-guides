# Recaption a Dataset

## Key Insight

Most public video ships with terrible labels — keyword spam, bare filenames, or [alt-text](/shared/glossary/#alt-text) that has nothing to do with the footage — and a generator can only learn the words-to-pixels mapping that its captions actually teach. Recaptioning replaces those with fresh, detailed descriptions written by a [VLM](/shared/glossary/#vlm) that actually watches each clip (these rewrites are called [synthetic captions](/shared/glossary/#synthetic-captions)), and it is widely considered the single highest-leverage change you can make to a video-training pipeline. This project recaptions 100k clips, trains a small [text-to-video (T2V)](/shared/glossary/#t2v) model on the original captions versus the rewritten ones, and measures the gap in how faithfully each follows a prompt — usually a large, cheap win for the same model and data.
