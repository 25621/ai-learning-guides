# Numeral Tokenization Audit

---

> If "1234" is four tokens in one model and one in another, arithmetic gets harder for free.

---

## Key Insight

[Tokenizers](/shared/glossary/#tokenizer) disagree on how to split numbers: some keep multi-digit chunks together, others break every digit apart. How digits are grouped directly shapes how hard arithmetic is for the model.

## Why This Matters

Early LLMs were notoriously bad at math partly because of inconsistent numeral tokenization. Auditing it reveals a concrete, fixable reason a model can fail at even simple sums.
