# Continued Pretraining

---

> Pick up an existing model and keep teaching it — without making it forget what it already knew.

---

## Key Insight

[Continued pretraining](/shared/glossary/#continued-pretraining) takes an open base model and trains it further on a specialized corpus (say, 1B tokens of medical or legal text). The experiment measures both the new capability gained and how much of the original ability is lost to [catastrophic forgetting](/shared/glossary/#catastrophic-forgetting).

## Why This Matters

Most teams will never pretrain from scratch, but many will adapt an existing model to their domain. Adding knowledge without erasing the base model's general skills is the practical core of [pretraining](/shared/glossary/#pretraining) work outside the frontier labs.
