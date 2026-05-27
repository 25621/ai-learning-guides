# Chat-Template Debugger

---

> If your training tokens and inference tokens differ by one space, the model quietly gets worse.

---

## Key Insight

A [chat template](/shared/glossary/#chat-template) wraps each turn in [special tokens](/shared/glossary/#special-tokens) (system, user, assistant). The model was fine-tuned on one exact byte pattern; rendering it by hand and byte-comparing surfaces any mismatch.

## Why This Matters

A single off-by-one space between training and inference is one of the most common — and hardest to spot — causes of a fine-tuned model silently underperforming.
