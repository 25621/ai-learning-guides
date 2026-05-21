# Manual Backprop

---

> Trust the autograd, but verify it by hand.

---

## Key Insight

Before relying entirely on [autograd](/shared/glossary/#autograd), it is crucial to compute the [gradients](/shared/glossary/#gradients) manually for a simple network. By applying the [chain rule](/shared/glossary/#chain-rule) step-by-step, you see exactly how the error signal flows backwards from the [loss](/shared/glossary/#loss-function) to the weights during the [backward pass](/shared/glossary/#backward-pass).

## Why This Matters

Writing manual backpropagation builds a strong intuitive foundation. When you understand the math behind the gradients, you can spot and fix [numerical issues](/shared/glossary/#numerical-issues), write more efficient custom operations, and truly grasp how deep learning models learn.