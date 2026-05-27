# BF16 vs FP8 Ablation

---

> Fewer bits per number means faster training — right up until the math stops being stable.

---

## Key Insight

This [ablation](/shared/glossary/#ablation) trains the same 100M model twice — once in [bfloat16](/shared/glossary/#bfloat16), once in [FP8](/shared/glossary/#fp8) — and compares the [loss](/shared/glossary/#loss-function) curves and training stability. Fewer bits per number save memory and run faster on modern GPUs, but leave less [headroom](/shared/glossary/#headroom) before [numerical issues](/shared/glossary/#numerical-issues) creep in.

## Why This Matters

Lower precision is how big GPUs get used efficiently, and FP8 is the 2024–2026 frontier for training speed. Knowing when dropping precision helps and when it quietly breaks a run is essential to training at scale without wasting hardware.
