# InstructPix2Pix-Style Data

## Key Insight

[InstructPix2Pix](/shared/glossary/#instructpix2pix) turns image editing into a single instruction — "make it winter" — by training on a synthetic dataset of (original image, instruction, edited image) triples. The clever part is *how that data is made*: a [large language model](/shared/glossary/#llm) like GPT invents an instruction and a pair of before/after captions, then a text-to-image model with [Prompt-to-Prompt](/shared/glossary/#prompt-to-prompt) generates a matched image pair that differs *only* in the edited detail. Generating that data yourself shows why the approach scales — no human ever hand-edits a photo — and why the resulting editor runs in one [forward pass](/shared/glossary/#forward-pass) instead of the slow per-image optimization that earlier editing methods required.
