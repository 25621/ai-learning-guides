# Speculative Agent Steps

---

> [Speculative decoding](/shared/glossary/#speculative-decoding) guesses the next *token*; this project guesses the next *action* — and rolls back when it guesses wrong.

---

## Key Insight

This project takes the guess-and-verify idea behind [speculative decoding](/shared/glossary/#speculative-decoding) and lifts it up a level: inside an [agent](/shared/glossary/#agent) loop, it speculatively runs the most likely next [tool call](/shared/glossary/#tool-call) *before* the model has finished deciding, then verifies the choice and rolls back the work if the guess was wrong. The agent's idle time — waiting on the model to think — becomes useful work done ahead of time.

## Why This Matters

Agent loops spend a lot of time stalled: waiting for the model to choose a tool, then waiting for that tool to return. Anywhere you can *guess and verify*, you can hide that latency, exactly as speculative decoding hides decode latency. This project generalizes the most reliable "free" trick in inference from single tokens to whole agent steps — a frontier the field is actively exploring.
