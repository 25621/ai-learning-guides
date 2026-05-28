# LLM-as-Judge Pipeline

---

> When no answer key exists, let a strong model be the grader.

---

## Key Insight

This project builds an [LLM-as-judge](/shared/glossary/#llm-as-judge) that reads a prompt and two candidate answers, votes for the better one, and then checks how often two judges (or the same judge run twice) agree — the way two restaurant critics might taste the same dish on the same day and you check how often they reach the same verdict; if Judge A says "Answer 1 is better" and Judge B says "Answer 2 is better" most of the time, neither score means very much, just as two critics who never agree make any single review unreliable.

## Why This Matters

Most real tasks — writing, summarizing, chatting — have no single correct answer to match against, so an LLM judge is the cheapest way to grade [open-ended](/shared/glossary/#open-ended) quality at scale, as long as you measure and correct for its biases.
