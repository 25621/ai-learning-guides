# Loss-Masking Bug Hunt

---

> Grade the model on its answers, not on the questions it was handed.

---

## Key Insight

This project runs [SFT](/shared/glossary/#sft) twice — once computing the [loss](/shared/glossary/#loss-function) over every token, and once with [loss masking](/shared/glossary/#loss-masking) so only the assistant's reply counts — and compares the two. When the prompt tokens are not masked, the model wastes effort learning to imitate questions instead of learning to answer them.

## Why This Matters

Loss masking is a one-line setting that is easy to get wrong and quietly degrades a fine-tune without ever throwing an error. Knowing how to spot its fingerprint saves you from a whole class of silent SFT bugs.
