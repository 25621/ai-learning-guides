# Implement AdamW from Scratch

---

> Adam remembers the past. AdamW forgets your weight.

---

## Key Insight

[AdamW](/shared/glossary/#adamw) is Adam with [weight decay](/shared/glossary/#weight-decay) applied directly to the parameters rather than to the gradients. It also uses [bias correction](/shared/glossary/#bias-correction) to counteract the zero-initialization of the first and second moment estimates, which would otherwise cause very small steps at the start of training.

## Why This Matters

AdamW is the default optimizer for most modern language and vision models. Implementing it by hand makes the update rule concrete and clarifies why decoupled weight decay outperforms L2 regularization — a distinction that matters especially when training transformers.
