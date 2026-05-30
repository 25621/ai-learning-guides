# Tune `k`

---

> Guess too few tokens and you barely speed up; guess too many and most of the work is wasted.

---

## Key Insight

This project sweeps `k`, the number of tokens the [draft model](/shared/glossary/#draft-model) proposes per step, and plots how the [acceptance rate](/shared/glossary/#acceptance-rate), speedup, and [tail latency](/shared/glossary/#tail-latency) change — finding the "knee" where adding more guesses stops helping.

## Why This Matters

A larger `k` means more tokens can be accepted per step, but also more wasted draft work whenever a guess is wrong, so there is a sweet spot. Measuring the whole curve teaches you to choose `k` for your own workload instead of copying a number from a blog post.
