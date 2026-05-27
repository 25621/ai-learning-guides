# SFT a 1B Base Model

---

> The first pass that turns a brilliant autocomplete into something that answers you.

---

## Key Insight

This project takes an open [base model](/shared/glossary/#base-model) and runs [supervised fine-tuning (SFT)](/shared/glossary/#sft) on instruction-response examples, then scores the result on [MT-Bench](/shared/glossary/#mt-bench). SFT does not teach new facts — it teaches the model the chat format and the habit of replying to a request instead of just continuing the text.

## Why This Matters

SFT is the first and cheapest step that turns a raw [next-token predictor](/shared/glossary/#next-token-prediction) into something that follows instructions. Almost every assistant you have used started with an SFT pass like this one.
