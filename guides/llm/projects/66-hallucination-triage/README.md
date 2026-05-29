# Hallucination Triage

---

> The bug is not what the model says; it's what it says when it should say nothing.

---

## Key Insight

This project builds a 100-prompt evaluation made of questions the model genuinely cannot know — invented names, future events, made-up acronyms — and [triages](/shared/glossary/#triage) the responses by how often the model responsibly says "I don't know" versus confidently inventing an answer ([hallucination](/shared/glossary/#hallucination)).

## Why This Matters

A model can ace knowledge [benchmarks](/shared/glossary/#benchmark) and still mislead users in production because the training objective rewards fluent continuation, not honest abstention; measuring the confident-wrong rate alongside the refusal rate is the only way to see this failure mode clearly before your users do.
