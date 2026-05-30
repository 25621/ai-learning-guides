# Medusa Heads

---

> Skip the second model — teach the big one to guess its own next few words.

---

## Key Insight

This project trains a few extra prediction [heads](/shared/glossary/#heads) ([Medusa](/shared/glossary/#eagle--medusa)) onto the [target model](/shared/glossary/#target-model) itself, so it proposes its own next tokens instead of running a separate [draft model](/shared/glossary/#draft-model); it then compares the [acceptance rate](/shared/glossary/#acceptance-rate) against using an external draft.

## Why This Matters

A separate draft model is one more thing to load, fit in memory, and keep in sync; self-speculation heads avoid all of that and usually accept more guesses, because they share the target's own knowledge. It is often the most practical way to get speculative speedups in production.
