# GCG Jailbreak

---

> Optimize a string of token gibberish until even a safety-tuned model agrees to break its rules.

---

## Key Insight

This project reproduces a small [GCG](/shared/glossary/#gcg) attack — a gradient-based search that finds an adversarial suffix to append to a prompt so that an aligned [open model](/shared/glossary/#open-model) produces unsafe output — and traces what the optimization is actually doing token by token.

## Why This Matters

GCG-style attacks demonstrate that safety training is brittle: a suffix found against one [open model](/shared/glossary/#open-model) often transfers to other models, including closed APIs; understanding the optimization is the basis for any serious [jailbreak](/shared/glossary/#jailbreak) defense or evaluation.
