# Custom Eval

---

> The only benchmark that matters is the one shaped like your problem.

---

## Key Insight

This project defines a 100-prompt evaluation that mirrors a specific [application](/shared/glossary/#application) — say, the customer-support chatbot for your online store — collects three independent grades per prompt, and measures how noisy those grades are. Think of it like building a custom road test for your own car instead of trusting a magazine's generic review: the 100 prompts look like the questions your real users actually ask, and grading each one three times tells you how much of the final score is real signal versus random fluctuation from the grader — the same way checking a thermometer three times at the same spot reveals how much you should trust any single reading.

## Why This Matters

Public [benchmarks](/shared/glossary/#benchmark) rarely match what your users actually do, and grading is noisy, so a small targeted eval with repeated grades tells you far more about real quality than a famous leaderboard number.
