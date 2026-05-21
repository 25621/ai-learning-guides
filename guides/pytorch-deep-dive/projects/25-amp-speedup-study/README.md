# AMP Speedup Study

---

> Half the bits, twice the speed — and usually the same answer.

---

## Key Insight

[Automatic mixed precision](/shared/glossary/#amp) runs most operations in 16-bit floats — [float16](/shared/glossary/#float16) or [bfloat16](/shared/glossary/#bfloat16) — instead of [float32](/shared/glossary/#float32), which halves memory traffic and unlocks fast [Tensor Cores](/shared/glossary/#tensor-core). float16 needs a [GradScaler](/shared/glossary/#gradscaler) to avoid [underflow](/shared/glossary/#underflow); bfloat16 shares float32's range and does not.

## Why This Matters

Mixed precision often gives a 2–3× speedup for one or two lines of code, with little or no loss in final accuracy — one of the highest-return changes you can make to a training script.
