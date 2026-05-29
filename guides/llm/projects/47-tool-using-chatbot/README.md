# Tool-Using Chatbot

---

> Give the model a calculator and it stops guessing at arithmetic.

---

## Key Insight

This project gives a chat model two tools — a calculator and a search function — through [function calling](/shared/glossary/#function-calling): the model emits a structured request, an external program runs it, and the result is fed back so the model can finish its answer.

## Why This Matters

Tools let an LLM do what it is bad at on its own — exact arithmetic, looking up fresh facts — turning a closed-book guesser into a system that can reach out to the outside world.
