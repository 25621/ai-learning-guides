# Minimal RAG

---

> Don't make the model memorize your documents — let it look them up.

---

## Key Insight

This project builds the simplest possible [RAG](/shared/glossary/#rag) pipeline: encode 1,000 Wikipedia paragraphs with a [sentence-embedding](/shared/glossary/#sentence-embedding) model, store the vectors, and at query time fetch the few closest paragraphs and paste them into the prompt before the model answers.

## Why This Matters

RAG is how you get an LLM to answer questions about *your* data — private or recent documents it never saw during training — without the cost of retraining it.
