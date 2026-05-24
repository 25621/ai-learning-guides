# Custom Vocab Extension

---

> Teaching a model a new "alphabet" means adding tokens and growing its [embedding](/shared/glossary/#embedding) table to match.

---

## Key Insight

Adding tokens to a [tokenizer](/shared/glossary/#tokenizer) requires resizing the model's [embedding matrix](/shared/glossary/#embedding-matrix) so every new ID gets a vector. Those new rows start untrained, so the model must learn what they mean.

## Why This Matters

Specialized notations — like the SMILES strings that describe molecules — often tokenize poorly out of the box. Extending the [vocabulary](/shared/glossary/#vocabulary) cleanly, without breaking the model's existing English, is a practical fine-tuning skill.
