# Train a 10M-Parameter LM

---

> The whole training loop fits on one screen — train it until none of it feels like magic.

---

## Key Insight

A 10-million-[parameter](/shared/glossary/#parameters) [language model](/shared/glossary/#llm) trained on a tiny Shakespeare file is the smallest honest [pretraining](/shared/glossary/#pretraining) run. The model is too small to be useful, which is the point: it is small enough to read every line of the loop and watch the [next-token-prediction](/shared/glossary/#next-token-prediction) objective at work.

## Why This Matters

Watching your own [loss](/shared/glossary/#loss-function) curve fall, on your own machine, removes the mystery from the whole field. Every billion-dollar training run is this same loop — forward pass, loss, backward pass, optimizer step — scaled up.
