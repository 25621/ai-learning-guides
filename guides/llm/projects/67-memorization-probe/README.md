# Memorization Probe

---

> Give the right opening line and the model finishes the page word for word.

---

## Key Insight

This project feeds an open model the first few sentences of documents from its known [pretraining](/shared/glossary/#pretraining) corpus and counts how often the model continues with the verbatim original text — direct evidence of [memorization](/shared/glossary/#memorization).

## Why This Matters

LLMs reproduce a small but non-trivial fraction of their training data exactly, with consequences for copyright, privacy (leaked PII), and security; knowing how much your model memorizes is a precondition for any deployment that touches sensitive data and a check on whether your [deduplication](/shared/glossary/#deduplication) pipeline is doing its job.
