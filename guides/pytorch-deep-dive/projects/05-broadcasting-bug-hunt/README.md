# Broadcasting Bug Hunt

---

> The most dangerous bugs are the ones that don't crash.

---

## Key Insight

[Broadcasting](/shared/glossary/#broadcasting) automatically "stretches" tensors to match shapes. While powerful, it can silently apply math to tensors that weren't meant to interact.

## Why This Matters

Silent errors are the hardest to find. A broadcasting mistake won't give you an error message; it will just give you wrong results that can ruin your model's training.
